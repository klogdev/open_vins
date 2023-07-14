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
    // as we suppose to clone imu state then do update in each iteration
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

void ToyManager::track_image_and_update(const ov_core::CameraData &cam_msg){
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

    // If we do not have VIO initialization, then try_to_initialize
    // will collect all imu mssages feeded above in the constructor
    // and update the state->_imu and _cov for the initialization
    if (!is_initialized_vio) {
        is_initialized_vio = try_to_initialize(message);
        if (!is_initialized_vio) {
        double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
        PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
        return;
        }
    }

    // Call on our propagate and update function
    do_feature_propagate_update(message);
}

void ToyManager::do_feature_propagate_update(const ov_core::CameraData &cam_msg){
    // for the state propagation and landmarks bookkeeping
    // update&filter the state at the end of this fxn

    // Return if the camera measurement is older than the current time
    if (state->_timestamp > cam_msg.timestamp) {
        PRINT_WARNING(YELLOW "image received too old, unable to do anything (prop dt = %3f)\n" RESET,
                    (cam_msg.timestamp - state->_timestamp));
        return;

    // Propagate the state forward to the current update time 
    // (extrapolation via IMU messages/prediction via EoM)
    // Also augment it with a new clone
    // NOTE: if the state is already at the given time (can happen in sim)
    // NOTE: then no need to prop since we already are at the desired timestep
    if (state->_timestamp != cam_msg.timestamp) {
        propagator->propagate_and_clone(state, cam_msg.timestamp);
    }
    // timing after propagate the state
    rT3 = boost::posix_time::microsec_clock::local_time();
    }

    // _clones_IMU load the imu message from current (available) time to
    // the latest (brand new) time as the poses. 
    // if there are state more than 5, we could do multiview triangulation
    if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
        PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                    std::min(state->_options.max_clone_size, 5));
        return;
    }

    // Return if we unable to propagate via extrapolation
    if (state->_timestamp != cam_msg.timestamp) {
        PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
        PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, cam_msg.timestamp - state->_timestamp);
        return;
    }

    //==================================
    // feature bookkeeping to select SLAM features from MSCKF feature
    //==================================
    // select features into 3 categories: lost, msckf(marg) and slam
    std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
    // if the feature is not available after last update time, we remove it
    feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

    // mark the features available until last clone time to be marginalized
    // which is the feature available after _timestamp but first observed 
    // before the left side of the sliding window
    if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
        feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }

    // assume only use one sensor, i.e. cam0

    // We also need to make sure that the max tracks does not contain any lost features
    // see below: we need to select slam features from marg features based on the criterion of tracking length
    // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
    // i.e. the conflict between lost & marg features
    it1 = feats_lost.begin();
    while (it1 != feats_lost.end()) {
        if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {

        it1 = feats_lost.erase(it1);
        } else {
        it1++;
        }
    }

    std::vector<std::shared_ptr<Feature>> feats_maxtracks; //get candidate pool for the slam features
    it2 = feats_marg.begin();
    while (it2 != feats_marg.end()){
        bool is_max_tracked = false;
        // it2 is a pointer to ov_core::Feature
        // the timestamps is a map between camera id and list of observation time
        for (const auto &cam: (*it2)->timestamps){
            if ((int)cams.second.size() > state->_options.max_clone_size) {
                is_max_tracked = true;
                break;
            }     
        }

        // the feature reaches max_track is good to be a slam feature
        if (is_max_tracked){
            feats_maxtracks.push_back(*it2);
            feats_marg.erase(it2);
        }
        else{
            it2++;
        }
    }

    // Append a new SLAM feature if we have the room to do so 
    // (i.e. the state's feature_SLAM is not full)
    // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
    // check where we init startup_time ***
    if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
        (int)state->_features_SLAM.size() < state->_options.max_slam_features) {
        // Get the total amount to add, then the max amount that we can add given our marginalize feature array
        int amount_to_add = (state->_options.max_slam_features) - (int)state->_features_SLAM.size();
        int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
        // If we have at least 1 that we can add, lets add it!
        // Note: we remove them from the feat_marg array since we don't want to reuse information
        if (valid_amount > 0) {
        // insert the features from candidate pool (feats_maxtracks) to the end of feats_slam
        feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
        feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
        }
    }

    // _feature_SLAM is an SP to ov_core Landmarks, which only contains backend id,
    // uv coord and should_mard flag; where ov_core Features also has pose info

    // loop the state's feature_SLAM to complete feat_slam for the update
    for (std::pair<const size_t, ov_core::Landmark> &curr_landmark: state->_features_SLAM){
        std::shared_ptr<Feature> curr_feat = trackFEATS->get_feature_database()->get_feature(curr_landmark.second->_featid);
        if(curr_feat != nullptr)
            feats_slam.push_back(curr_feat);
        // we assume only one sensor and it've been init as cam0
        if (curr_feat == nullptr)
            curr_landmark->should_marg = true;
        if (curr_landmark.second->update_fail_count > 1)
            curr_landmark.second->should_marg = true;
    }

    // marg(remove) slam features with should_marg flag, which we will not care them in the future update
    // here marginalize is not related to the nullspace but 
    // here it means that the states of these elements will not be involved in future updates. 
    // However, their information that can affect the motion states has already been included through the previous steps
    // like a Markov model
    ov_msckf::StateHelper::marginalize_slam(state);

    // we now separate slam features into re-observed & new-triangulated
    // the re-observed one, i.e. already available in features_SLAM, we could directly use it
    // however for the new one, we need to have extra preprocess, i.e. delayed_init for the future use
    std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
    for (size_t i = 0; i < feats_slam.size(); i++){
        if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
        feats_slam_UPDATE.push_back(feats_slam.at(i));
        } else {
        feats_slam_DELAYED.push_back(feats_slam.at(i));
        }
    }

    // finally, wrap up MSCKF features (i.e. ones not being used for slam updates)
    // which 1. the lost features; 2. the marg features; 
    // 3. the candidate pool in the maxtrack but was not selected due to full of state vector
    std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
    featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
    featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

    //=============================
    // update&filter for the SLAM&MSCKF features
    //=============================
    

}



