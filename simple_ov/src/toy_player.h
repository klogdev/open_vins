#ifndef SIMPLE_OV_TOYPLAYER_H
#define SIMPLE_OV_TOYPLAYER_H

#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "feat/FeatureInitializerOptions.h"
#include "core/VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class TrackDescriptor;
class FeatureInitializer;
class FeatureDatabase;
class Feature;
class YamlParser;
}

namespace ov_type {
class Landmark;
}

namespace ov_init {
class InertialInitializer;
} 

namespace ov_msckf{
    class VioManager;
    class VioManagerOptions;
    class State;
    class StateHelper;
    class UpdaterMSCKF;
    class UpdaterSLAM;
    class Propagator;
}

namespace simple_ov {

/*
 *@brief: light version of the VIO system manager to handle initialization,
 *state propagate and filtering/update
*/
class ToyManager{
    public:
        ToyManager(ov_msckf::VioManagerOptions &options_);

        void feed_imu_data(const ov_core::ImuData &imu_msg);

        void feed_camera_data(const ov_core::CameraData &cam_msg);

    protected:
        /// Manager parameters
        ov_msckf::VioManagerOptions params;

        void track_image_and_update(const ov_core::CameraData &cam_msg);

        void do_feature_propagate_update(const ov_core::CameraData &cam_msg);

        /// State initializer
        std::shared_ptr<ov_init::InertialInitializer> initializer;

        /// Boolean if we are initialized or not
        bool is_initialized_vio = false;

        std::shared_ptr<ov_msckf::State> state;

        std::shared_ptr<ov_msckf::Propagator> propagator;

        /// Our sparse feature tracker (klt or descriptor)
        std::shared_ptr<ov_core::TrackBase> trackFEATS;

        /// Our MSCKF feature updater
        std::shared_ptr<ov_msckf::UpdaterMSCKF> updaterMSCKF;

        /// Our SLAM/ARUCO feature updater
        std::shared_ptr<ov_msckf::UpdaterSLAM> updaterSLAM;

        // Startup time of the filter
        double startup_time = -1;     

        // Good features that where used in the last update (used in visualization)
        std::vector<Eigen::Vector3d> good_features_MSCKF;

        boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    };
}  // namespace simple_ov
#endif // SIMPLE_OV_TOYPLAYER_H