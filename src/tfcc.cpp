#include <iostream>
#include <tensorflow/c/c_api.h>
#include <cppflow/cppflow.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <mavros_msgs/State.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>

float waypoint1[3] = {2.0, 0.5, 1.5}, waypoint2[3] = {4.0, 0.0, 1.0};
float current_position[3],current_position1[3],current_position2[3],current_velocity[3],last_velocity[3]={0,0,0},current_acceleration[3],current_rate[3];
float ROS_FREQ = 30.0;

// void stateCallback(const mavros_msgs::State::ConstPtr &msg)
// {
//   std::cout << (*msg).mode << std::endl;
// }

void visualCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
  current_position[0] = (*msg).pose.pose.position.x;
  current_position[1] = (*msg).pose.pose.position.y;
  current_position[2] = (*msg).pose.pose.position.z;
  current_position1[0] = waypoint1[0] - current_position[0];
  current_position1[1] = waypoint1[1] - current_position[1];
  current_position1[2] = waypoint1[2] - current_position[2];
  current_position2[0] = waypoint2[0] - current_position[0];
  current_position2[1] = waypoint2[1] - current_position[1];
  current_position2[2] = waypoint2[2] - current_position[2];
  current_velocity[0] = (*msg).twist.twist.linear.x;
  current_velocity[1] = (*msg).twist.twist.linear.y;
  current_velocity[2] = (*msg).twist.twist.linear.z;
  current_acceleration[0] = (current_velocity[0] - last_velocity[0]) * ROS_FREQ;
  current_acceleration[1] = (current_velocity[1] - last_velocity[1]) * ROS_FREQ;
  current_acceleration[2] = (current_velocity[2] - last_velocity[2]) * ROS_FREQ;
  current_rate[0] = (*msg).twist.twist.angular.x;
  current_rate[1] = (*msg).twist.twist.angular.y;
  current_rate[2] = (*msg).twist.twist.angular.z;
}

int main(int argc, char **argv) {

  // std::cout << a << std::endl;
  ros::init(argc, argv, "planner");
  ros::NodeHandle n;
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
  ros::Rate loop_rate(30);

  // ros::Subscriber state_sub = n.subscribe<mavros_msgs::State>("/uav0/mavros/state",
  //                                                             10,
  //                                                             stateCallback);
  ros::Subscriber position_sub = n.subscribe<nav_msgs::Odometry>("/uav0/mavros/local_position/odom",
                                                                  10,
                                                                  visualCallback);  
  cppflow::model model("/home/zhoujin/learning/model/model1");

  while (ros::ok())
  {
    auto input = cppflow::tensor({current_position1[0],current_position1[1],current_position1[2],
                                  current_position2[0],current_position2[1],current_position2[2],
                                  current_velocity[0],current_velocity[1],current_velocity[2],
                                  current_acceleration[0],current_acceleration[1],current_acceleration[2],
                                  current_rate[0],current_rate[1],current_rate[2]
                                  });
    input = cppflow::expand_dims(input,0);
    auto output = model({{"serving_default_dense_input:0", input}},{"StatefulPartitionedCall:0"});
    auto output1 = output[0].get_data<float>();
    std::cout << output1[0] << std::endl;
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}