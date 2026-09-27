#include "gn10_pointcloud_localization/floor_plane.hpp"
#include <stdexcept>
void require(bool ok){if(!ok)throw std::runtime_error("floor plane check failed");}
int main(){
 const float identity[12]={1,0,0,0,0,1,0,0,0,0,1,0};std::vector<float> cloud;
 for(int i=0;i<25;++i)for(int j=0;j<25;++j){float x=-5+i*.4f,y=-5+j*.4f;cloud.insert(cloud.end(),{x,y,.002f*x-.007f*y+.01f+float(.005*std::sin(i*19+j*7))});}
 for(int i=0;i<100;++i)cloud.insert(cloud.end(),{float(i%10)-5,float(i/10)-5,.14f});
 auto plane=gn10::estimateFloor(cloud,identity);require(plane.valid&&plane.support>500);
 require(std::abs(plane.a-.002)<.001&&std::abs(plane.b+.007)<.001&&std::abs(plane.c-.01)<.002);
 const float y=-5,x=0,z=.045;require(z>.04&&z-plane.a*x-plane.b*y-plane.c<.01);
 require(!gn10::estimateFloor({},identity).valid);
 std::vector<float> desk;for(int i=0;i<1000;++i)desk.insert(desk.end(),{float(i%25)*.05f,float(i/25)*.05f,.76f});
 require(!gn10::estimateFloor(desk,identity).valid);
 std::vector<float> small;for(int i=0;i<400;++i)small.insert(small.end(),{2.f+float(i%20)*.01f,2.f+float(i/20)*.01f,.02f});
 require(!gn10::estimateFloor(small,identity).valid);
}
