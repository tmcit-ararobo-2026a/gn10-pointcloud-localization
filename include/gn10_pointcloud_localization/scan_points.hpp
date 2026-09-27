#pragma once
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace gn10 {
struct ScanPoints { std::vector<float> xyz; std::vector<double> times; double start{0}, end{0}; bool valid{false}, timed{false}; };
// MID360 timestamp: absolute nanoseconds (FLOAT64), or relative offset_time (UINT32 ns).
// Preserve row padding. Never infer point times from array order.
inline ScanPoints decodeScan(const sensor_msgs::msg::PointCloud2& msg) {
    ScanPoints out; out.start=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9; out.end=out.start;
    const sensor_msgs::msg::PointField *x=nullptr,*y=nullptr,*z=nullptr,*t=nullptr;
    bool absolute=false;
    for (const auto& f:msg.fields) {
        if(f.count!=1) continue;
        if(f.datatype==sensor_msgs::msg::PointField::FLOAT32 && f.offset+4<=msg.point_step) {
            if(f.name=="x")x=&f;
            if(f.name=="y")y=&f;
            if(f.name=="z")z=&f;
        }
        if(f.name=="timestamp" && f.datatype==sensor_msgs::msg::PointField::FLOAT64 && f.offset+8<=msg.point_step) {t=&f;absolute=true;}
    }
    if(!t) for(const auto& f:msg.fields) if(f.name=="offset_time" && f.datatype==sensor_msgs::msg::PointField::UINT32 && f.count==1 && f.offset+4<=msg.point_step)t=&f;
    if(!x||!y||!z||msg.is_bigendian||!msg.point_step||msg.row_step<static_cast<size_t>(msg.width)*msg.point_step||msg.data.size()<static_cast<size_t>(msg.row_step)*msg.height)return out;
    out.timed=t!=nullptr;
    const double start_ns=static_cast<double>(msg.header.stamp.sec)*1e9+msg.header.stamp.nanosec;
    for(uint32_t r=0;r<msg.height;++r)for(uint32_t c=0;c<msg.width;++c) {
        auto p=msg.data.data()+static_cast<size_t>(r)*msg.row_step+static_cast<size_t>(c)*msg.point_step;
        float a,b,d;std::memcpy(&a,p+x->offset,4);std::memcpy(&b,p+y->offset,4);std::memcpy(&d,p+z->offset,4);
        if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(d))continue;
        double dt=0;
        if(t){if(absolute){double ns;std::memcpy(&ns,p+t->offset,8);dt=(ns-start_ns)*1e-9;}else{uint32_t ns;std::memcpy(&ns,p+t->offset,4);dt=ns*1e-9;}
            // Reject inconsistent clocks/units rather than quietly mixing scans.
            if(!std::isfinite(dt)||dt < -1e-6 || dt>0.2)return ScanPoints{};
        }
        double stamp=out.start+std::max(0.0,dt);out.end=std::max(out.end,stamp);
        out.xyz.insert(out.xyz.end(),{a,b,d});out.times.push_back(stamp);
    }
    out.valid=!out.xyz.empty();return out;
}
}
