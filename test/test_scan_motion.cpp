#include "gn10_pointcloud_localization/motion_history.hpp"
#include "gn10_pointcloud_localization/scan_points.hpp"
#include "gn10_pointcloud_localization/custom_scan.hpp"
#include "gn10_pointcloud_localization/custom_cloud_conversion.hpp"
#include <stdexcept>
#include <iostream>
void require(bool v) {if(!v)throw std::runtime_error("scan motion check failed");}
int main(){
 gn10::MotionHistory h;require(h.add({1,0,0,3.13}));require(h.add({1.1,0.1,0,-3.13}));
 auto p=h.at(1.05);require(p&&std::abs(p->x-.05)<1e-6&&std::abs(std::abs(p->yaw)-3.14159265)<1e-5);
 require(!h.at(.99)&&!h.at(1.11)&&!h.add({1.05,0,0,0}));
 gn10::TimedPose a{1,0,0,0}, b{1.1,.1,0,0};float x=2,y=1;
 gn10::MotionHistory::transformPoint(a,b,x,y);require(std::abs(x-1.9)<1e-5&&y==1);
 auto predicted=gn10::MotionHistory::advance({1,5,3,1.57079632679},a,b);
 require(std::abs(predicted.x-5)<1e-5&&std::abs(predicted.y-3.1)<1e-5);
 float rx=1,ry=0;gn10::MotionHistory::transformPoint(a,{1.1,0,0,1.57079632679},rx,ry);
 require(std::abs(rx)<1e-5&&std::abs(ry+1)<1e-5);
 gn10::MotionHistory gap;gap.add(a);gap.add({1.4,0,0,0});require(!gap.at(1.2));
 sensor_msgs::msg::PointCloud2 msg;msg.header.stamp.sec=1;msg.width=1;msg.height=2;msg.point_step=24;msg.row_step=32;msg.data.resize(64);
 for(const auto& v:std::vector<std::pair<std::string,int>>{{"x",0},{"y",4},{"z",8}}){sensor_msgs::msg::PointField f;f.name=v.first;f.offset=v.second;f.datatype=7;f.count=1;msg.fields.push_back(f);}
 sensor_msgs::msg::PointField t;t.name="timestamp";t.offset=16;t.datatype=8;t.count=1;msg.fields.push_back(t);
 for(int i=0;i<2;++i){float x=2,y=0,z=1;double ns=1e9+i*1e8;auto d=msg.data.data()+i*32;std::memcpy(d,&x,4);std::memcpy(d+4,&y,4);std::memcpy(d+8,&z,4);std::memcpy(d+16,&ns,8);}
 auto s=gn10::decodeScan(msg);require(s.valid&&s.timed&&s.xyz.size()==6&&std::abs(s.end-1.1)<1e-6);
 livox_ros_driver2::msg::CustomMsg custom;custom.header=msg.header;custom.timebase=1000000000;custom.point_num=2;
 for(int i=0;i<2;++i){livox_ros_driver2::msg::CustomPoint p;p.x=2;p.y=0;p.z=1;p.offset_time=i*100000000;custom.points.push_back(p);}
 auto native=gn10::decodeScan(custom);require(native.valid&&native.timed&&native.xyz==s.xyz&&native.times==s.times&&native.end==s.end);
 auto display=gn10::customToPointCloud2(custom);auto decoded_display=gn10::decodeScan(display);
 require(display.header==custom.header&&decoded_display.xyz==native.xyz&&decoded_display.times==native.times&&display.width==2&&display.point_step==24);
 custom.point_num=1;require(!gn10::decodeScan(custom).valid);custom.point_num=2;
 custom.timebase+=10000000;require(!gn10::decodeScan(custom).valid);custom.timebase=1000000000;
 custom.points.back().offset_time=300000000;require(!gn10::decodeScan(custom).valid);
 double bad=1e9+1e9;std::memcpy(msg.data.data()+48,&bad,8);require(!gn10::decodeScan(msg).valid);
 msg.fields.pop_back();s=gn10::decodeScan(msg);require(s.valid&&!s.timed);
 msg.row_step=1;require(!gn10::decodeScan(msg).valid);
 std::cout<<"scan timing, padding, interpolation and deskew checks passed\n";
}
