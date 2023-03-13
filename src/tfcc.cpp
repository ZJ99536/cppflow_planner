#include <iostream>
#include <tensorflow/c/c_api.h>
#include <cppflow/cppflow.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <mavros_msgs/State.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>

float waypoint1[3] = {2.0, 0.5, 1.5}, waypoint2[3] = {4.0, 0.0, 1.0};
float current_position[3],current_position1[3],current_position2[3];
float current_velocity[3],last_velocity[3]={0,0,0};
float current_acceleration[3],current_rate[3];
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
  current_acceleration[2] = (current_velocity[2] - last_velocity[2]) * ROS_FREQ + 9.81;
  current_rate[0] = (*msg).twist.twist.angular.x;
  current_rate[1] = (*msg).twist.twist.angular.y;
  current_rate[2] = (*msg).twist.twist.angular.z;
  last_velocity[0] = current_velocity[0];
  last_velocity[1] = current_velocity[1];
  last_velocity[2] = current_velocity[2];
}

int main(int argc, char **argv) {

  // std::cout << a << std::endl;
  ros::init(argc, argv, "planner");
  ros::NodeHandle n;

  geometry_msgs::Vector3Stamped position_setpoint,velocity_setpoint,acceleration_setpoint,rate_setpoint,command_setpoint;
  ros::Publisher position_pub = n.advertise<geometry_msgs::Vector3Stamped>("/planner/position", 1);
  ros::Publisher velocity_pub = n.advertise<geometry_msgs::Vector3Stamped>("/planner/velocity", 1);
  ros::Publisher attitude_pub = n.advertise<geometry_msgs::Vector3Stamped>("/planner/attitude", 1);
  ros::Publisher rate_pub = n.advertise<geometry_msgs::Vector3Stamped>("/planner/rate", 1);  
  ros::Publisher command_pub = n.advertise<geometry_msgs::Vector3Stamped>("/planner/command", 1);  

  ros::Rate loop_rate(ROS_FREQ);

  // ros::Subscriber state_sub = n.subscribe<mavros_msgs::State>("/uav0/mavros/state",
  //                                                             10,
  //                                                             stateCallback);
  ros::Subscriber position_sub = n.subscribe<nav_msgs::Odometry>("/uav0/mavros/local_position/odom",
                                                                  10,
                                                                  visualCallback);  
  cppflow::model model("/home/zhoujin/learning/model/model1");
  cppflow::model val("/home/zhoujin/learning/model/val1");

  while (ros::ok())
  {
    auto input = cppflow::tensor({current_position1[0],current_position1[1],current_position1[2],
                                  current_position2[0],current_position2[1],current_position2[2],
                                  current_velocity[0],current_velocity[1],current_velocity[2],
                                  current_acceleration[0],current_acceleration[1],current_acceleration[2],
                                  current_rate[0],current_rate[1],current_rate[2]
                                  });
    auto input_val = cppflow::tensor({current_position1[0],current_position1[1],current_position1[2],
                                      current_position2[0],current_position2[1],current_position2[2]
                                      });
    input = cppflow::expand_dims(input,0);
    input_val = cppflow::expand_dims(input_val,0);
    auto output = model({{"serving_default_dense_input:0", input}},{"StatefulPartitionedCall:0"});
    auto output_val = val({{"serving_default_dense_input:0", input_val}},{"StatefulPartitionedCall:0"});
    auto output1 = output[0].get_data<float>();
    auto output1_val = output_val[0].get_data<float>();
    
    position_setpoint.vector.x = ((waypoint2[0] - output1[3]) + (waypoint1[0] - output1[0])) / 2;
    position_setpoint.vector.y = ((waypoint2[1] - output1[4]) + (waypoint1[1] - output1[1])) / 2;
    position_setpoint.vector.z = ((waypoint2[2] - output1[5]) + (waypoint1[2] - output1[2])) / 2;
    position_setpoint.header.stamp = ros::Time::now();
    position_pub.publish(position_setpoint);

    velocity_setpoint.vector.x = output1[6];
    velocity_setpoint.vector.y = output1[7];
    velocity_setpoint.vector.z = output1[8];
    velocity_setpoint.header.stamp = ros::Time::now();
    velocity_pub.publish(velocity_setpoint);

    acceleration_setpoint.vector.x = output1[9];
    acceleration_setpoint.vector.y = output1[10];
    acceleration_setpoint.vector.z = output1[11];
    acceleration_setpoint.header.stamp = ros::Time::now();
    attitude_pub.publish(acceleration_setpoint);

    rate_setpoint.vector.x = output1[12];
    rate_setpoint.vector.y = output1[13];
    rate_setpoint.vector.z = output1[14];
    rate_setpoint.header.stamp = ros::Time::now();
    rate_pub.publish(rate_setpoint);

    command_setpoint.vector.x = 0;
    command_setpoint.vector.y = output1_val[0];
    command_setpoint.vector.z = output1_val[1];
    command_setpoint.header.stamp = ros::Time::now();
    command_pub.publish(command_setpoint);

    std::cout << output[0] << std::endl;

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}