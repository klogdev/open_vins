#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "toy_player.h"

namespace simple_ov {

}  // namespace simple_ov

ToyManager::ToyManager(ov_msckf::VioManagerOptions &options_){
    this->params = options_;

    state = std::make_shared<ov_msckf::State>(params.state_options);

    Eigen::VectorXd cam_imu_dt;
    cam_imu_dt.resize(1);
    cam_imu_dt(0) = params.calib_camimu_dt;

    //value:current best estimation; fej: first estimation using smooth
    //_calib is an sp to the type Vec, which contains both curr& estimate state vec
    state->_calib_dt_CAMtoIMU->set_value(cam_imu_dt);
    state->_calib_dt_CAMtoIMU->set_fej(cam_imu_dt);

    //Loop through and load each of the cameras
    //here the vector entry for cam intrinsic and pose are separate vectors
    state->_cam_intrinsics_cameras = params.camera_intrinsics;
    for (int i = 0; i < state->_options.num_cameras; i++) {
        state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
        state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
        state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
        state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
    }

    //assume we only have one camera
    int init_max_features = params.init_options.init_max_features;
    //assume only use descriptors; instantiate a new TrackDescriptor object,
    //which is derived from TrackBase; have method feed_new_camera(image)&perform_detection
    //contains a member var: database_ to store all features
    trackFEATS = std::shared_ptr<ov_core::TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_params.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));

    // Initialize state propagator, for the implementation of the EoM
    propagator = std::make_shared<ov_msckf::Propagator>(params.imu_noises, params.gravity_mag);

    // intialize the initializer, this will simly implement fee_imu, 
    // the detailed initialization will be implemented under track_image_update
    // which calls the initializer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

    // Initialize the updaters for the bookkeeping; we now ignore the handling of zero update
    updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
    updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);
}

ToyManager::feed_imu_data(const ov_core::ImuData &imu_msg){
    // the latest cloned time/to be marginalized time
    double oldest_time = state->margtimestep();
    // _timestamp is the latest updated step, if it smaller than oldest
    // means we are not initialized yet (no cloned state available)
    if(oldest_time > state->_timestamp){
        oldest_time = -1;
    }
    // we need to trace the time of initialization with a delay between
    // cam to imu; also manually backward 0.1 for safe
    if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
    }
    // push imu messages to both propagator & initializer anyway;
    // if not init yet, the propagator will early return when try to prop
    propagator->feed_imu(message, oldest_time);

    if (!is_initialized_vio) {
        initializer->feed_imu(message, oldest_time);
    }
}

ToyManager::track_image_and_update(const ov_core::CameraData &cam_msg){
    // Start timing for the later debugging
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Assert we have valid measurement data and ids
    // i.e. the data from sensor is not empty; each sensor have one image
    // no repeated sensor messages
    assert(!cam_msg.sensor_ids.empty());
    assert(cam_msg.sensor_ids.size() == cam_msg.images.size());
    for (size_t i = 0; i < cam_msg.sensor_ids.size() - 1; i++) {
        assert(cam_msg.sensor_ids.at(i) != cam_msg.sensor_ids.at(i + 1));
    }

    // Downsample if needed; pyr is pyramid downsampling
    // that can keep most of features and descriptors
    ov_core::CameraData message = cam_msg;
    for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
        cv::Mat img = message.images.at(i);
        cv::Mat mask = message.masks.at(i);
        cv::Mat img_temp, mask_temp;
        cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
        message.images.at(i) = img_temp;
        cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
        message.masks.at(i) = mask_temp;
    }

    // send camera message to the feature tracking
    // it will do feature detection & matching & update the feature DB
    trackFEATS->feed_new_camera(message);

    // timing after handling the new features
    rT2 = boost::posix_time::microsec_clock::local_time();
}
