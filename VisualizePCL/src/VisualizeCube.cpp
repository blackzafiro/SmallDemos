#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

void set_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cpu_cloud,
               int x, int y, int z, uint8_t r, uint8_t g, uint8_t b)
{
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    cpu_cloud->points[y * cpu_cloud->width + x].x = x * 100;
    cpu_cloud->points[y * cpu_cloud->width + x].y = y * 100;
    cpu_cloud->points[y * cpu_cloud->width + x].z = z * 100;
    cpu_cloud->points[y * cpu_cloud->width + x].rgb = rgb;
}

int main(int argc, char *argv[])
{
    // Create empty point cloud and ptr to it.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cpu_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    cpu_cloud->width = 2;
    cpu_cloud->height = 2;
    cpu_cloud->is_dense = true;
    
    // Add 8 points
    cpu_cloud->points.resize(8);
    set_color(cpu_cloud, 0, 0, 0, 200, 100, 100);
    set_color(cpu_cloud, 0, 1, 0, 100, 200, 100);
    set_color(cpu_cloud, 1, 0, 0, 100, 100, 200);
    set_color(cpu_cloud, 1, 1, 0, 100, 200, 100);
    set_color(cpu_cloud, 0, 0, 1, 100, 100, 200);
    set_color(cpu_cloud, 0, 1, 1, 100, 200, 100);
    set_color(cpu_cloud, 1, 0, 1, 100, 100, 200);
    set_color(cpu_cloud, 1, 1, 1, 100, 200, 100);
    
    // Create viewer (there are no threads here)
    pcl::visualization::CloudViewer viewer("Kinect V2");
    viewer.showCloud(cpu_cloud);
	
    // To see the cloud instead of the red, green, black rectangles use:
    std::cout << "Press R to center, use mouse to rotate." << std::endl;

    while(!viewer.wasStopped()){}
    
    return 0;    
}
