#include "gn10_pointcloud_localization/surface_refinement.hpp"
#include <stdexcept>
void require(bool ok) {if(!ok) throw std::runtime_error("surface refinement failed");}
int main() {
 std::vector<FieldObject> map{{BOX,0,3,0,2,4,.1},{BOX,4,0,0,2,.1,3}};
 const PoseCandidate truth{.4f,-.7f,.3f};std::vector<float> cloud;
 const double c=std::cos(truth.yaw),s=std::sin(truth.yaw);
 auto point=[&](double x,double y){double dx=x-truth.x,dy=y-truth.y;cloud.insert(cloud.end(),{float(c*dx+s*dy),float(-s*dx+c*dy),1.f});};
 for(int i=0;i<200;++i){point(-3.+6.*i/200,2.9);point(3.9,-2.+4.*i/200);}
 // Unmapped moving returns must not influence the surface fit.
 for(int i=0;i<100;++i)point(1.+i*.001,0.);
 PoseCandidate fitted;require(gn10::refineSurfaces(cloud,map,.08,{.42f,-.72f,.305f},fitted));
 require(std::hypot(fitted.x-truth.x,fitted.y-truth.y)<.001&&std::abs(fitted.yaw-truth.yaw)<.001);
 std::vector<float> wall;for(int i=0;i<200;++i)wall.insert(wall.end(),{float(-3.+6.*i/200),2.9f,1.f});
 require(!gn10::refineSurfaces(wall,{map.front()},.08,{0,0,0},fitted));
 double r;Eigen::Vector2d n;
 require(gn10::surfaceResidual({CYLINDER,0,0,0,2,1,0},1.02,0,1,r,n)&&std::abs(r-.02)<1e-6&&n.x()==1);
 require(!gn10::surfaceResidual(map.front(),0,3,3,r,n));
 return 0;
}
