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
    trackFEATS = std::shared_ptr<ov_core::TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_params.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));

    // Initialize state propagator, for the implementation of the EoM
    propagator = std::make_shared<ov_msckf::Propagator>(params.imu_noises, params.gravity_mag);

    // Our state initialize
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
}
