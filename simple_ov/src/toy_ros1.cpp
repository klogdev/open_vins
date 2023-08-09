#include <ros/ros.h>
#include <memory>

#include "toy_player.h"
#include "ROS1Visualizer.h" //should be replaced by customized viz later
#include "opencv_yaml_parse.h"

std::shared_ptr<ToyManager> sys; //apps
std::shared_ptr<ROS1Visualizer> viz; //encalpsulated pipeline for publisher & callback

int main(int argc, char **argv){
    std::string config_path;
    
    if(argc < 2)
        std::cout << " " << argv[0] << "need configuration path"

    config_path = argv[1];

    // Launch our ros node
    ros::init(argc, argv, "ros1_serial_msckf");
    auto nh = std::make_shared<ros::NodeHandle>("~");
    // node's params that controls ros server, here it set the "config_path" key
    // to the given config_path
    nh->param<std::string>("config_path", config_path, config_path); 

    // Load the config
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);

    // Verbosity
    std::string verbosity = "DEBUG";
    parser->parse_config("verbosity", verbosity);

    
}