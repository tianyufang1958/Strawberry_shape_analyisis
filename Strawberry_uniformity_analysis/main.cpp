#include <iostream>
#include <boost/thread/thread.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/features/principal_curvatures.h>
#include <boost/thread/thread.hpp>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef struct PCURVATURE{
    int index;
    float curvature;
}PCURVATURE;


typedef pcl::PointXYZRGB PointType;
string slice_name_side, slice_name_top;

vector<string> listFile(char folder_name[])
{
    DIR *pDIR;
    struct dirent *entry;
    vector<string> files;
    if( pDIR=opendir(folder_name) ){
            while(entry = readdir(pDIR)){
                    if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
                   // cout << entry->d_name << "\n";

                    files.push_back(entry->d_name);
                  //  string folder_name = "Michelle_pic/";
            }
            closedir(pDIR);
            std::sort(files.begin(), files.end() );
    }
    return files;
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (255, 255, 255);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "sample cloud");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (0.0);
 // viewer->removeCoordinateSystem (0.0);

  viewer->resetCamera();
  return (viewer);
}

void addboundingbox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float &height, float &length, float &width, int plot)
{
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*cloud, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZRGB minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    // Final transform
    const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
    const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

    pcl::PointXYZRGB minPoint1, maxPoint1;
    pcl::getMinMax3D(*cloud, minPoint1, maxPoint1);

    height = maxPoint.z - minPoint.z;
    length = maxPoint.y - minPoint.y;
    width = maxPoint.x - minPoint.x;

    if (plot == 1)
    {

    pcl::visualization::PCLVisualizer *visu;
    visu = new pcl::visualization::PCLVisualizer();

    visu->setBackgroundColor (255, 255, 255);
    visu->setCameraPosition(0, 0, 1, 0, 0, 0);
    visu->addPointCloud<pcl::PointXYZRGB>(cloud, "bboxedCloud");
    visu->addCoordinateSystem (0);
    visu->addCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, "bbox");
    //visu->addCube( minPoint1.x, maxPoint1.x, minPoint1.y,  maxPoint1.y, minPoint1.z, maxPoint1.z);
    visu->setShapeRenderingProperties ( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, "bbox");
    visu->resetCamera();

    while(!visu->wasStopped())
    {
      visu->spinOnce (10);

    }
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pre_process (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, int low,  int high)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    for (int i = 0; i < cloud_in->points.size(); i++)
    {
        if (cloud_in->points[i].b>low && cloud_in->points[i].b<high)
        {
            inliers->indices.push_back(i);
        }
    }

    pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true);
    eifilter.setInputCloud (cloud_in);
    eifilter.setIndices (inliers);
    eifilter.filter (*cloud_out);


    return cloud_out;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr TM_origin(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float& height, float& length, float& width)
{
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //校正主方向间垂直
    eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
    eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));

    Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tm_inv = Eigen::Matrix4f::Identity();
    tm.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();   //R.
    tm.block<3, 1>(0, 3) = -1.0f * (eigenVectorsPCA.transpose()) *(pcaCentroid.head<3>());//  -R*t
    tm_inv = tm.inverse();

    pcl::PointCloud<PointType>::Ptr transformedCloud(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*cloud, *transformedCloud, tm);

    PointType min_p1, max_p1;
    Eigen::Vector3f c1, c;
    pcl::getMinMax3D(*transformedCloud, min_p1, max_p1);
    c1 = 0.5f*(min_p1.getVector3fMap() + max_p1.getVector3fMap());

    height = max_p1.z - min_p1.z;
    length = max_p1.y - min_p1.y;
    width =  max_p1.x - min_p1.x;


    Eigen::Affine3f tm_inv_aff(tm_inv);
    pcl::transformPoint(c1, c, tm_inv_aff);

    Eigen::Vector3f whd, whd1;
    whd1 = max_p1.getVector3fMap() - min_p1.getVector3fMaphttp://dict.cn/lame();
    whd = whd1;
    float sc1 = (whd1(0) + whd1(1) + whd1(2)) / 3;  //点云平均尺度，用于设置主方向箭头大小

    const Eigen::Quaternionf bboxQ1(Eigen::Quaternionf::Identity());
    const Eigen::Vector3f    bboxT1(c1);

    const Eigen::Quaternionf bboxQ(tm_inv.block<3, 3>(0, 0));
    const Eigen::Vector3f    bboxT(c);

    //变换到原点的点云主方向
    PointType op;
    op.x = 0.0;
    op.y = 0.0;
    op.z = 0.0;
    Eigen::Vector3f px, py, pz;
    Eigen::Affine3f tm_aff(tm);
    pcl::transformVector(eigenVectorsPCA.col(0), px, tm_aff);
    pcl::transformVector(eigenVectorsPCA.col(1), py, tm_aff);
    pcl::transformVector(eigenVectorsPCA.col(2), pz, tm_aff);
    PointType pcaX;
    pcaX.x = sc1 * px(0);
    pcaX.y = sc1 * px(1);
    pcaX.z = sc1 * px(2);
    PointType pcaY;
    pcaY.x = sc1 * py(0);
    pcaY.y = sc1 * py(1);
    pcaY.z = sc1 * py(2);
    PointType pcaZ;
    pcaZ.x = sc1 * pz(0);
    pcaZ.y = sc1 * pz(1);
    pcaZ.z = sc1 * pz(2);

    //初始点云的主方向
    PointType cp;
    cp.x = pcaCentroid(0);
    cp.y = pcaCentroid(1);
    cp.z = pcaCentroid(2);
    PointType pcX;
    pcX.x = sc1 * eigenVectorsPCA(0, 0) + cp.x;
    pcX.y = sc1 * eigenVectorsPCA(1, 0) + cp.y;
    pcX.z = sc1 * eigenVectorsPCA(2, 0) + cp.z;
    PointType pcY;
    pcY.x = sc1 * eigenVectorsPCA(0, 1) + cp.x;
    pcY.y = sc1 * eigenVectorsPCA(1, 1) + cp.y;
    pcY.z = sc1 * eigenVectorsPCA(2, 1) + cp.z;
    PointType pcZ;
    pcZ.x = sc1 * eigenVectorsPCA(0, 2) + cp.x;
    pcZ.y = sc1 * eigenVectorsPCA(1, 2) + cp.y;
    pcZ.z = sc1 * eigenVectorsPCA(2, 2) + cp.z;

    //visualization
  /*  pcl::visualization::PCLVisualizer viewer;

   // pcl::visualization::PointCloudColorHandlerCustom<PointType> tc_handler(transformedCloud, 0, 255, 0); //转换到原点的点云相关
    viewer.addPointCloud(transformedCloud,  "transformCloud");
    viewer.addCube(bboxT1, bboxQ1, whd1(0), whd1(1), whd1(2), "bbox1");
    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox1");
    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "bbox1");

    viewer.addArrow(pcaX, op, 1.0, 0.0, 0.0, false, "arrow_X");
    viewer.addArrow(pcaY, op, 0.0, 1.0, 0.0, false, "arrow_Y");
    viewer.addArrow(pcaZ, op, 0.0, 0.0, 1.0, false, "arrow_Z");

    viewer.addCoordinateSystem(0.5f*sc1);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }*/
  return transformedCloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr outlier_remove(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int meank, int std)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (meank);
    sor.setStddevMulThresh (std);
    sor.filter (*cloud_filter);

    return cloud_filter;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr holder_extraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr holder(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_HSV(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloudXYZRGBtoXYZHSV(*cloud, *cloud_HSV);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    for (int i = 0; i < cloud->points.size(); i++)                         //holder
    {
        if (cloud_HSV->points[i].h < 250  && cloud_HSV->points[i].h > 180)            // increase the first parameter to have better visualisation of holder
        {
            inliers->indices.push_back(i);
        }
    }
    pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true);
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (inliers);
    eifilter.filter (*holder);

    return holder;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr body_extraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr body(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_HSV(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloudXYZRGBtoXYZHSV(*cloud, *cloud_HSV);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    for (int i = 0; i < cloud->points.size(); i++)                         //holder
    {
        if ((cloud_HSV->points[i].h < 30 && cloud_HSV->points[i].h > 0)|| cloud_HSV->points[i].h > 320)            // increase the first parameter to have better visualisation of holder
        {
            inliers->indices.push_back(i);
        }
    }
    pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true);
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (inliers);
    eifilter.filter (*body);

    return body;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr holder_body_extraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_HSV(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_str(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointCloudXYZRGBtoXYZHSV(*cloud, *cloud_HSV);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    for (int i = 0; i < cloud->points.size(); i++)                        //body with ackene
    {
        if ((cloud_HSV->points[i].h < 250  && cloud_HSV->points[i].h > 180) || (cloud_HSV->points[i].h < 15 || cloud_HSV->points[i].h > 320))
        {
            inliers->indices.push_back(i);
        }
    }
    pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true);
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (inliers);
    eifilter.filter (*cloud_str);

    return cloud_str;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cap_body_extraction(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_HSV(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_str(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointCloudXYZRGBtoXYZHSV(*cloud, *cloud_HSV);

    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    for (int i = 0; i < cloud->points.size(); i++)                        //body with ackene
    {
        if (cloud_HSV->points[i].h < 75 || cloud_HSV->points[i].h > 320)
        {
            inliers->indices.push_back(i);
        }
    }


    pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true);
    eifilter.setInputCloud (cloud);
    eifilter.setIndices (inliers);
    eifilter.filter (*cloud_str);

    return cloud_str;
}

pcl::PolygonMesh poisson_mesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int plot)
{

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNumberOfThreads (8);
    ne.setInputCloud (cloud);
    ne.setRadiusSearch (0.5);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid (*cloud, centroid);
    ne.setViewPoint (centroid[0], centroid[1], centroid[2]);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
    ne.compute (*cloud_normals);

    for (size_t i = 0; i < cloud_normals->size (); ++i)
      {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
      }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_smoothed_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
    pcl::concatenateFields (*cloud, *cloud_normals, *cloud_smoothed_normals);

    pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
    poisson.setDepth (9);
    poisson.setInputCloud(cloud_smoothed_normals);
    pcl::PolygonMesh mesh;
    poisson.reconstruct (mesh);

    if (plot ==1)
    {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (255, 255, 255);
        viewer->addPolygonMesh(mesh,"meshes",0);
        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        //viewer->addPointCloud<pcl::PointXYZRGB> (cloud, cloud_normals, "sample cloud");
        //viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, cloud_normals, 10, 0.05, "normals");
        viewer->resetCamera();
        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
        }
    }

    return mesh;
}

float signedVolumeOfTriangle(pcl::PointXYZ p1, pcl::PointXYZ p2, pcl::PointXYZ p3)
{
    float v321 = p3.x*p2.y*p1.z;
    float v231 = p2.x*p3.y*p1.z;
    float v312 = p3.x*p1.y*p2.z;
    float v132 = p1.x*p3.y*p2.z;
    float v213 = p2.x*p1.y*p3.z;
    float v123 = p1.x*p2.y*p3.z;
    return (1.0f/6.0f)*(-v321 + v231 + v312 - v132 - v213 + v123);
}

float calculate_vol(pcl::PolygonMesh mesh)                                                      //Volume Calculation
{
    float vol = 0.0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud,*cloud);
    for(int triangle=0;triangle<mesh.polygons.size();triangle++)
    {
        pcl::PointXYZ pt1 = cloud->points[mesh.polygons[triangle].vertices[0]];
        pcl::PointXYZ pt2 = cloud->points[mesh.polygons[triangle].vertices[1]];
        pcl::PointXYZ pt3 = cloud->points[mesh.polygons[triangle].vertices[2]];
        vol += signedVolumeOfTriangle(pt1, pt2, pt3);
    }
    return fabs(vol);
}

double AreaOfTriangle(pcl::PointXYZ p1, pcl::PointXYZ p2, pcl::PointXYZ p3)
{
    double ax = p2.x - p1.x;
    double ay = p2.y - p1.y;
    double az = p2.z - p1.z;
    double bx = p3.x - p1.x;
    double by = p3.y - p1.y;
    double bz = p3.z - p1.z;
    double cx = ay*bz - az*by;
    double cy = az*bx - ax*bz;
    double cz = ax*by - ay*bx;

    return 0.5 * sqrt(cx*cx + cy*cy + cz*cz);
}

float calculate_area(pcl::PolygonMesh mesh)                                                      //Volume Calculation
{
    float area = 0.0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud,*cloud);
    for(int triangle=0;triangle<mesh.polygons.size();triangle++)
    {
        pcl::PointXYZ pt1 = cloud->points[mesh.polygons[triangle].vertices[0]];
        pcl::PointXYZ pt2 = cloud->points[mesh.polygons[triangle].vertices[1]];
        pcl::PointXYZ pt3 = cloud->points[mesh.polygons[triangle].vertices[2]];
        area += AreaOfTriangle(pt1, pt2, pt3);
    }

    return area;
}

Mat remove_small_region(Mat img_thresh)
{
     dilate(img_thresh, img_thresh, getStructuringElement(MORPH_RECT, Size(11, 11)));
     erode(img_thresh, img_thresh, getStructuringElement(MORPH_RECT, Size(11, 11)));
    // imshow("thresh", img_thresh);
    // waitKey(0);
     Mat img_thresh_copy = img_thresh.clone();
     vector<vector<Point> > contours;
     findContours(img_thresh_copy, contours, RETR_CCOMP, CHAIN_APPROX_NONE);

     for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
     {
         if (it->size() > 200  )
         {
             it=contours.erase(it);
         }
         else
             ++it;
     }
     drawContours(img_thresh, contours, -1, Scalar(0), FILLED);

     return img_thresh;
}

void fillEdgeImage(Mat edgesIn, Mat& filledEdgesOut)
{
    Mat edgesNeg = edgesIn.clone();

    floodFill(edgesNeg, Point(0,0), CV_RGB(255,255,255));
    bitwise_not(edgesNeg, edgesNeg);
    filledEdgesOut = (edgesNeg | edgesIn);

    return;
}

void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));https://www.ebay.co.uk/sh/ord/?filter=status:ALL_ORDERS
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
}
double getOrientation(const vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(img, cntr, p1, Scalar(0, 255, 100), 2);
    drawAxis(img, cntr, p2, Scalar(255, 100, 0), 4);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x)*180/M_PI + (atan2(eigen_vecs[0].y, eigen_vecs[0].x) > 0 ? 0 : 360); // orientation in radians
    if (angle > 270)
        angle = angle - 360 + 20;
    else
        angle = angle + 20;
  //  cout<<"y "<<eigen_vecs[0].y<<" x "<< eigen_vecs[0].x<<endl;
    return angle;
}

vector<PCURVATURE> getModelCurvatures(pcl::PolygonMesh mesh)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);

    //计算法线--------------------------------------------------------------------------------------
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree); //设置搜索方法
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    //ne.setRadiusSearch(0.05); //设置半径邻域搜索
    ne.setKSearch(15);
    ne.compute(*cloud_normals); //计算法向量
    //计算法线--------------------------------------------------------------------------------------
    //计算曲率-------------------------------------------------------------------------------------
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures>pc;
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr cloud_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    pc.setInputCloud(cloud);
    pc.setInputNormals(cloud_normals);
    pc.setSearchMethod(tree);
   // pc.setRadiusSearch(0.05);
    pc.setKSearch(5);
    pc.compute(*cloud_curvatures);

    //获取曲率集
    vector<PCURVATURE> tempCV;
    pcl::PointXYZ tempPoint;
    float curvature = 0.0;
    PCURVATURE pv;
    tempPoint.x = tempPoint.y = tempPoint.z=0.0;
    for (int i = 0; i < cloud_curvatures->size();i++)
    {
        //平均曲率
       // curvature = ((*cloud_curvatures)[i].pc1 + (*cloud_curvatures)[i].pc2) / 2;
        //高斯曲率
        curvature = (*cloud_curvatures)[i].pc1 * (*cloud_curvatures)[i].pc2;
        //pv.cPoint = tempPoint;
        pv.index = i;
        pv.curvature = curvature;
        tempCV.insert(tempCV.end(),pv);
    }


   /* pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor (0, 0, 0);
   // viewer.addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, 10, 3, "cloud_normals");
    viewer.addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, cloud_curvatures);
     while (!viewer.wasStopped ())
    {
         viewer.spinOnce ();
    }*/

        return tempCV;
}


vector<float> side_slice_analysis(pcl::PointCloud<pcl::PointXYZRGB>::Ptr body_cloud_filter)
{
    Mat slice ((1+4)*200, ((int)(sqrt(5))+4)*200, CV_8UC3, Scalar(0, 0, 0));
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
    vector<float> circularities;
    vector<float> areas;
    vector<float> areas_thresh;
    vector<float> angles;

    pcl::PointXYZRGB minPt, maxPt;
    pcl::getMinMax3D(*body_cloud_filter, minPt, maxPt);
    float body_height = abs(maxPt.z - minPt.z);

    vector<Point> arear;
    for(int i = 0; i < 100; i++)
    {
        float theta = (float)i * 2*M_PI/(float)100;
       // cout<<"theta "<<M_PI<<endl;
       // transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

        Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
        transform_1 (0,0) = std::cos (theta);
        transform_1 (0,1) = -sin(theta);
        transform_1 (1,0) = sin (theta);
        transform_1 (1,1) = std::cos (theta);

        pcl::transformPointCloud (*body_cloud_filter, *transformed_cloud, transform_1);

        for (int j = 0; j < transformed_cloud->size(); j++)
        {
          double x = transformed_cloud->points[j].x;
          double y = transformed_cloud->points[j].y;
          double z = transformed_cloud->points[j].z;

          //cout<<(int)(z+3)*100<<" "<<(int)(x+3)*100<<" "<<endl;
          if(z < (maxPt.z - 0.25*body_height) && z > (minPt.z + 0.25*body_height))
          {
              slice.at<Vec3b>((int)((z+3)*100), (int)((x+3)*100))[0] = 255;
              slice.at<Vec3b>((int)((z+3)*100), (int)((x+3)*100))[1] = 255;
              slice.at<Vec3b>((int)((z+3)*100), (int)((x+3)*100))[2] = 255;
          }
          else
          {
              slice.at<Vec3b>((int)((z+3)*100), (int)((x+3)*100))[0] = 0;
              slice.at<Vec3b>((int)((z+3)*100), (int)((x+3)*100))[1] = 0;
              slice.at<Vec3b>((int)((z+3)*100), (int)((x+3)*100))[2] = 255;
          }
        }

        Mat gray_output, threshold_out;

    //    vector<vector<Point> > contours;
      //  Mat threshold_output =  remove_small_region(threshold_out);

        Mat gray, bw, bw_filled;;
        cvtColor(slice, gray, COLOR_BGR2GRAY);
        threshold(gray, bw, 20, 255, THRESH_BINARY | THRESH_OTSU);

        float area_thresh = 0;
        for (int m = 0; m < slice.rows; m++)
        {
            for (int n = 0; n < slice.rows; n++)
            {
                if (bw.at<uchar>(m,n) == 255)
                {
                    arear.push_back(Point(n,m));
                    area_thresh = area_thresh + 1;
                }
            }
        }
        areas_thresh.push_back(area_thresh);

        vector<vector<Point> > hull(1);
        convexHull( Mat(arear), hull[0], false);
        float perimeter = arcLength(hull[0], true);
        float area = contourArea(hull[0]);
        areas.push_back(area);
        float circularity = 12.57 * area / (perimeter*perimeter);
        circularities.push_back(circularity);

        Mat drawing = Mat::zeros(slice.size(), CV_8UC3);
        Scalar color = Scalar(255, 0, 0);
        for (int m = 0; m < hull.size(); m++)
        {
            drawContours( slice, hull, m, color, 4);
            drawContours( bw, hull, m, Scalar(255), 2);
        }

        fillEdgeImage(bw, bw_filled);

        vector<vector<Point> > contours;
        findContours(bw_filled, contours, RETR_LIST, CHAIN_APPROX_NONE);
        float angle = getOrientation(contours[0], slice);
       // cout<<"angle = "<<angle<<endl;
        angles.push_back(angle);

 /*       if (i == 0)
        {
            imwrite(slice_name_side+"_0.jpg", slice);
        }

        if (i == 25)
        {
            imwrite(slice_name_side+"_1.jpg", slice);
        }

        if (i == 50)
        {
            imwrite(slice_name_side+"_2.jpg", slice);
        }

        if (i == 75)
        {
            imwrite(slice_name_side+"_3.jpg", slice);
        }*/

  //     imshow("display", slice);
 //       waitKey(0);

        slice.setTo(Scalar(0));
        arear.clear();
    }

    int max_area = *max_element(areas.begin(), areas.end());
    int min_area = *min_element(areas.begin(), areas.end());

 //   cout<<"max area "<<max_area<<"; min area "<<min_area<<"; ratio "<<(float)max_area/min_area<<endl;

    float ave_circularity = accumulate(circularities.begin(), circularities.end(), 0.0)/ circularities.size();
    float ave_area_thresh = accumulate(areas_thresh.begin(), areas_thresh.end(), 0.0)/ areas_thresh.size();
    float ave_angle = accumulate(angles.begin(), angles.end(), 0.0)/ angles.size();
    float sd_circularity, sd_area_thresh, sd_angle;
    float sum_circularity_2 = 0;
    float sum_area_thresh_2 = 0;
    float sum_angle_2 = 0;
    for (int c = 0; c < circularities.size(); c++)
    {
        sum_circularity_2 += circularities[c]*circularities[c];
        sum_area_thresh_2 += areas_thresh[c]*areas_thresh[c];
        sum_angle_2 += angles[c]*angles[c];
    }
    sd_circularity = sum_circularity_2/circularities.size() - ave_circularity*ave_circularity;
    sd_area_thresh = sum_area_thresh_2/areas_thresh.size() - ave_area_thresh*ave_area_thresh;
    sd_angle = sum_angle_2/angles.size() - ave_angle*ave_angle;

   // float CV_circularity = sqrt(sd_circularity) / ave_circularity;
    float CV_area_thresh = sqrt(sd_area_thresh) / ave_area_thresh;
    float CV_angle = sqrt(sd_angle) / ave_angle;

    float ratio_area = (float)max_area/min_area;  

    vector<float> output;
    output.push_back(sqrt(CV_area_thresh));                                                  //vector<float> (ave_circularity, ratio_area)
   // output.push_back(sqrt(CV_circularity));
    output.push_back(ratio_area);
    output.push_back(CV_angle);
    areas.clear();

    return output;
}

vector<float> top_slice_analysis(pcl::PointCloud<pcl::PointXYZRGB>::Ptr body_cloud_filter)
{
    Mat slice ((1+4)*150, (1+4)*150, CV_8UC3, Scalar(255, 255, 255));
    //vector<float> diameters;
    vector<float> length_rect;
    vector<float> width_rect;
    vector<float> output;
    vector<float> circularities;
    vector<float> areas;
    vector<Point3f> mc3ds;

    pcl::PointXYZRGB minPt, maxPt;
    pcl::getMinMax3D(*body_cloud_filter, minPt, maxPt);

    float step = (maxPt.z - minPt.z) / 100;
    float body_height = abs(maxPt.z - minPt.z);

    pcl::PolygonMesh mesh = poisson_mesh(body_cloud_filter, 0);
    vector<PCURVATURE> allCurvates;
 //   allCurvates = getModelCurvatures(mesh);


    for (int i = 0; i < 98; i++)
    {
        for (int j = 0; j < body_cloud_filter->size(); j++)
        {
           double x = body_cloud_filter->points[j].x;
           double y = body_cloud_filter->points[j].y;
           double z = body_cloud_filter->points[j].z;

           if (abs(z - (maxPt.z - i*step)) < 0.03)
           {
               slice.at<Vec3b>((int)((x+3)*100), (int)((y+3)*100))[0] = 255;
               slice.at<Vec3b>((int)((x+3)*100), (int)((y+3)*100))[1] = 255;
               slice.at<Vec3b>((int)((x+3)*100), (int)((y+3)*100))[2] = 0;
           }
        }

        Mat gray_output, threshold_output;
        vector<Point> arear;
     //   vector<vector<Point> > contours;

        cvtColor(slice, gray_output, COLOR_RGB2GRAY);
        threshold(gray_output, threshold_output, 20, 255, THRESH_BINARY );
        for (int i = 0; i < slice.rows; i++)
        {
            for (int j = 0; j < slice.rows; j++)
            {
                if (threshold_output.at<uchar>(i,j) == 255)
                {
                    arear.push_back(Point(j,i));
                }
            }
        }

        vector<vector<Point> > hull(1);
        convexHull( Mat(arear), hull[0], false);
        Moments mu;
        mu = moments(hull[0]);
        Point2f mc;

        Point3f mc3d;
        //add 1e-5 to avoid division by zero
        mc = Point2f( static_cast<float>(mu.m10 / (mu.m00 + 1e-5)), static_cast<float>(mu.m01 / (mu.m00 + 1e-5)) );
        for (int m = 0; m < hull.size(); m++)
        {
            drawContours(threshold_output, hull, m, Scalar(255), 2);
            drawContours(slice, hull, m, Scalar(0,255,0), 2);
        }
       // circle( threshold_output, mc, 4, Scalar(255), -1 );
        mc3d.x = mc.x/100-3;
        mc3d.y = mc.y/100-3;
        mc3d.z = maxPt.z - i*step;
        mc3ds.push_back(mc3d);

        float perimeter = arcLength(hull[0], true);
        float area = contourArea(hull[0]);

        areas.push_back(area);
        float circularity = 12.57 * area / (perimeter*perimeter);
        circularities.push_back(circularity);

        vector<RotatedRect> minRect(1);
        vector<RotatedRect> minEllipse(1);

        if (arear.size() > 0)
        {
            minRect[0] = minAreaRect( Mat(arear) );

            Scalar color = Scalar(0, 0, 255);

            Point2f rect_points[4];
            minRect[0].points( rect_points );
            for( int j = 0; j < 4; j++ )
                  line( slice, rect_points[j], rect_points[(j+1)%4], color, 3, 8 );

            rect_points[0].x = rect_points[0].x/100-3;
            rect_points[1].x = rect_points[1].x/100-3;
            rect_points[0].y = rect_points[0].y/100-3;
            rect_points[1].y = rect_points[1].y/100-3;
            rect_points[2].x = rect_points[2].x/100-3;
            rect_points[2].y = rect_points[2].y/100-3;

            float size1 = sqrt((rect_points[0].y-rect_points[1].y)*(rect_points[0].y-rect_points[1].y) + (rect_points[0].x-rect_points[1].x)*(rect_points[0].x-rect_points[1].x));
            float size2 = sqrt((rect_points[1].y-rect_points[2].y)*(rect_points[1].y-rect_points[2].y) + (rect_points[1].x-rect_points[2].x)*(rect_points[1].x-rect_points[2].x));
            if (size1 > size2)
            {
                length_rect.push_back(size1);
                width_rect.push_back(size2);
            }
            else
            {
                length_rect.push_back(size2);
                width_rect.push_back(size1);
            }
        }
        else
        {
            length_rect.push_back(0);
            width_rect.push_back(0);
        }

    //    imshow("Display", slice);
     //   waitKey(0);

        if (i == 20)
        {
            imwrite(slice_name_top+"20.jpg", slice);
        }

        if (i == 40)
        {
            imwrite(slice_name_top+"max.jpg", slice);
        }

        if (i == 60)
        {
            imwrite(slice_name_top+"60.jpg", slice);
        }

        if (i == 80)
        {
            imwrite(slice_name_top+"80.jpg", slice);
        }

        slice.setTo(Scalar(0,0,0));
    }

    int max_area_index = max_element(areas.begin(), areas.end()) - areas.begin();


    float max_area = *max_element(areas.begin(), areas.end());
    float max_slice_circularity0 = circularities[max_area_index];
    float max_slice_circularity1 = circularities[max_area_index-10];
    float max_slice_circularity2 = circularities[max_area_index+10];
    float ave_circularity = (max_slice_circularity0+max_slice_circularity1+max_slice_circularity2)/3;

    int max_length_index = max_element(length_rect.begin(), length_rect.end()) - length_rect.begin();
    float max_length_rect = *max_element(length_rect.begin(), length_rect.end());
    float min_width_rect = width_rect[max_length_index];

    cout<<"max index = "<<max_length_index<<endl;

    float ratio_length_width = max_length_rect/min_width_rect;

    float ideal_centroid_length = sqrt((mc3ds[9].x - mc3ds[89].x)*(mc3ds[9].x - mc3ds[89].x)+(mc3ds[9].y - mc3ds[89].y)*(mc3ds[9].y - mc3ds[89].y)+(mc3ds[9].z - mc3ds[89].z)*(mc3ds[9].z - mc3ds[89].z));
    float actual_centroid_length = 0;

    for (int g = 9; g<= 89; g++)
    {
        float length = sqrt((mc3ds[g].x - mc3ds[g-1].x)*(mc3ds[g].x - mc3ds[g-1].x)+(mc3ds[g].y - mc3ds[g-1].y)*(mc3ds[g].y - mc3ds[g-1].y)+(mc3ds[g].z - mc3ds[g-1].z)*(mc3ds[g].z - mc3ds[g-1].z));
        actual_centroid_length += length;
    }
    float ratio_actual_ideal = actual_centroid_length / ideal_centroid_length;
  //  cout<<"ideal = "<<ideal_centroid_length<<" actual "<<actual_centroid_length<<endl;

    output.push_back(ratio_length_width);                                                  // vector<float> (ratio_length_height, ratio_width_height, max_length_index, max length/15%, max length/85%)
    output.push_back(ave_circularity);
    output.push_back(ratio_actual_ideal);

    float x_top = mc3ds[5].x;
    float y_top = mc3ds[5].y;
    float z_top = mc3ds[5].z;
    float x_bot = mc3ds[95].x;
    float y_bot = mc3ds[95].y;
    float z_bot = mc3ds[95].z;

    Eigen::Vector4f top(x_top, y_top, z_top, 0.0);
    Eigen::Vector4f bot(x_bot, y_bot, z_bot, 0.0);

    //vtkSmartPointer<vtkDataSet> data = pcl::visualization::createLine (top, bot);

    for (int j = 0; j < body_cloud_filter->size(); j++)
    {
       double x = body_cloud_filter->points[j].x;
       double y = body_cloud_filter->points[j].y;
       double z = body_cloud_filter->points[j].z;

       if (abs(z - (maxPt.z - 40*step)) < 0.02)
       {
            body_cloud_filter->points[j].r = 255;
            body_cloud_filter->points[j].g = 255;
            body_cloud_filter->points[j].b = 255;
       }

       if (abs(z - (maxPt.z - 20*step)) < 0.02)
       {
            body_cloud_filter->points[j].r = 255;
            body_cloud_filter->points[j].g = 255;
            body_cloud_filter->points[j].b = 0;
       }

       if (abs(z - (maxPt.z - 60*step)) < 0.02)
       {
            body_cloud_filter->points[j].r = 0;
            body_cloud_filter->points[j].g = 255;
            body_cloud_filter->points[j].b = 255;
       }

       if (abs(z - (maxPt.z - 80*step)) < 0.02)
       {
            body_cloud_filter->points[j].r = 0;
            body_cloud_filter->points[j].g = 0;
            body_cloud_filter->points[j].b = 255;
       }
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
   // viewer->addLine(top, bot, "line");
    viewer->addPointCloud(body_cloud_filter);
    viewer->addCoordinateSystem (1.0);
    viewer->resetCamera();
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    }

    return output;
}


int main()
{
 char* shapes[10] = { "temp", "1_extra", "2_extra", "3", "4", "5", "6", "7", "8", "9"};
 vector<string> shapes_all (shapes, shapes+10);
 char c_str[40] = "uniform_no_holder/";
 string slash = "/";

 ofstream file;
 file.open ("uniform2409.txt", ios::app);

 for (int m = 0; m < 1; m++)
  {
     char* full_text = new char[strlen(c_str)+strlen(shapes[0])+1];
     strcpy(full_text, c_str );
     strcat(full_text, shapes[m]);

     vector<string> cloud_names = listFile(full_text);
     cout<<full_text<<endl;

 for (int i = 0; i < cloud_names.size(); i++)
 {
    slice_name_side = "temp_side_top_images/" +shapes_all[m] +slash+ cloud_names[i] + "-side.jpg";
    slice_name_top = "temp_side_top_images/" + shapes_all[m] +slash+ cloud_names[i] +  "-top.jpg";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
   // string file_name = c_str + slash + cloud_names[i];
    string file_name = full_text + slash + cloud_names[i];
    cout<<file_name<<endl;
    if (pcl::io::loadPLYFile (file_name, *cloud) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file \n");
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr body_cloud;
    body_cloud = body_extraction(cloud);                                                                    //segment body
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr body_cloud_filter = outlier_remove(body_cloud, 300, 3);

    vector<float> cir_area = side_slice_analysis(body_cloud_filter);
    cout<<shapes[m]<<"  "<<cloud_names[i]<< ", Area CV: "<<cir_area[0]<<", Area ratio: "<<cir_area[1]<<" Angle CV "<<cir_area[2]<<endl;

    vector<float> dim_ratio_index = top_slice_analysis(body_cloud_filter);
    cout<<"Ratio of length and width: "<<dim_ratio_index[0]<<" Average circularity: "<<dim_ratio_index[1]<<" Centroid length: "<<dim_ratio_index[2]<<endl<<endl;

    file<<cloud_names[i]<<","<<shapes[m]<<","<<cir_area[0]<<","<<cir_area[1]<<","<<cir_area[2]<<","<<","<<dim_ratio_index[0]<<","<<dim_ratio_index[1]<<","<<dim_ratio_index[2]<<endl;

  /*  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = simpleVis(body_cloud_filter);

    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    }*/
 }
 }
 file.close();
}
