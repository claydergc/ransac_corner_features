#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/io/png_io.h>
#include <pcl/common/common.h>

#include <iostream>
#include <ctime>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <pcl/common/distances.h>
#include <pcl/keypoints/iss_3d.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/filters/extract_indices.h>

#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <tf/transform_datatypes.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>

#include <limits>

using namespace Eigen;
using namespace std;
using namespace pcl;
using namespace pcl::search;

//catkin_make -DCMAKE_BUILD_TYPE=Release

//this version attempt to draw features from corner_extraction and loam_velodyne node


//const unsigned int MIN_CLUSTER_POINTS = 200;
const unsigned int MIN_CLUSTER_POINTS = 70;
const unsigned int MAX_CLUSTER_POINTS = 150;
//const unsigned int MIN_CLUSTER_POINTS = 200;
//const unsigned int MIN_CLUSTER_POINTS = 150;
//const unsigned int MIN_CLUSTER_POINTS = 400;
//const double CLUSTER_DISTANCE = 0.15;
const double CLUSTER_DISTANCE = 0.5;
//const double CLUSTER_DISTANCE = 0.05;
//const double CLUSTER_DISTANCE = 0.05;
//const double CLUSTER_DISTANCE = 0.5;
//const double CLUSTER_DISTANCE = 0.7;
//const double GROUND_SEGMENTATION_DISTANCE = 0.7;
const double GROUND_SEGMENTATION_DISTANCE = 1.3;

ros::Publisher pub1;
ros::Publisher pub2;

PointCloud<PointXYZ>::Ptr globalCloud (new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr cornerFeatures (new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr loamSharp (new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr cornersMerged(new PointCloud<PointXYZ>);

PointCloud<InterestPoint> featureCloud;

PointCloud<PointXYZ>::Ptr cornersAnt[12];

PointCloud<PointXYZ> cornersGlobal;
PointCloud<PointXYZ> featuresCloudPerLaserTotal[12];

bool isOdometryEnabled = false;


/*Incluye la información de los datos para formar la recta y los puntos que encontró RANSAC*/
typedef struct linePair
{
  Eigen::VectorXf line;
  PointCloud<PointXYZ> lineCloud; 
}LinePair;

void extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector< pcl::PointCloud<PointXYZ> > cloudVector);

unsigned int frame=0;

Eigen::Affine3f transGlobal;
Eigen::Affine3f transOdom;

template <typename PointT>
void fromPCLPointCloud2ToVelodyneCloud(const pcl::PCLPointCloud2& msg, pcl::PointCloud<PointT>& cloud1D, std::vector< pcl::PointCloud<PointT> >& cloudVector, unsigned int rings)
{
  cloud1D.header   = msg.header;
  cloud1D.width    = msg.width;
  cloud1D.height   = msg.height;
  cloud1D.is_dense = msg.is_dense == 1;
  uint32_t num_points = msg.width * msg.height;
  cloud1D.points.resize (num_points);
  uint8_t* cloud_data1 = reinterpret_cast<uint8_t*>(&cloud1D.points[0]);
  
  pcl::PointCloud<PointT>* cloudPerLaser = new pcl::PointCloud<PointT>[rings];
  uint8_t* cloud_data2[rings];

  unsigned int pointsCounter[rings] = {0};

  for(unsigned int i=0; i<rings; ++i)
  {
    cloudPerLaser[i] = pcl::PointCloud<PointT>();
    cloudPerLaser[i].header   = msg.header;
    cloudPerLaser[i].width    = msg.width;
    cloudPerLaser[i].height   = msg.height;
    cloudPerLaser[i].is_dense = msg.is_dense == 1;
    cloudPerLaser[i].points.resize (num_points);
    cloud_data2[i] = reinterpret_cast<uint8_t*>(&cloudPerLaser[i].points[0]);
  }

  for (uint32_t row = 0; row < msg.height; ++row)
  {
    const uint8_t* row_data = &msg.data[row * msg.row_step];
      
    for (uint32_t col = 0; col < msg.width; ++col)
    {
        const uint8_t* msg_data = row_data + col * msg.point_step;

        //float* x = (float*)msg_data;
        //float* y = (float*)(msg_data + 4);
        //float* z = (float*)(msg_data + 8);
        //float* i = (float*)(msg_data + 16);
        uint16_t* ring = (uint16_t*)(msg_data+20);
        memcpy (cloud_data2[*ring], msg_data, 22);
        memcpy (cloud_data1, msg_data, 22);
        pointsCounter[*ring]++;
        cloud_data1 += sizeof (PointT);
        cloud_data2[*ring] += sizeof (PointT);
    }
  }

  cloudVector = std::vector< pcl::PointCloud<PointT> >(rings);

  for(unsigned int i=0; i<rings; ++i)
  {
      cloudPerLaser[i].width = pointsCounter[i];
      cloudPerLaser[i].height = 1;
      cloudPerLaser[i].points.resize (pointsCounter[i]);
      cloudVector[i] = (cloudPerLaser[i]);
  }

  delete[] cloudPerLaser;
}

void cloud_callback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	vector< pcl::PointCloud<pcl::PointXYZ> > cloudVector;
	pcl::PCLPointCloud2 pcl_pc2;
	pcl_conversions::toPCL(*cloud_msg, pcl_pc2);

	fromPCLPointCloud2ToVelodyneCloud (pcl_pc2, *cloud, cloudVector, 16);

	//pcl::fromROSMsg(*cloud_msg , *cloud);
	cout<<"Frame "<<frame<<endl;
	extractFeatures(cloud, cloudVector);
	frame++;
}

//Callback de features LOAM
void cloud_callback2 (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg(*cloud_msg , *cloud);
  pcl::transformPointCloud (*cloud, *cloud, transOdom);
  *loamSharp += *cloud;
}


void odom_callback (const nav_msgs::Odometry::ConstPtr& odom_msg)
{
  isOdometryEnabled = true;
  ROS_INFO("Seq: [%d]", odom_msg->header.seq);
  ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z);
  
  transOdom = Eigen::Affine3f::Identity();
  transOdom.translation() << odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z;
  transOdom.rotate(Eigen::Quaternionf(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x, odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z));

  pcl::transformPointCloud (*cornersMerged, cornersGlobal, transOdom);
  pcl::transformPointCloud (featureCloud, featureCloud, transOdom);
}

void clusterize(PointCloud<PointXYZ>::Ptr cloud, vector< PointCloud<PointXYZ> >& clusters, double threshold, double minClusterPoints, double maxClusterPoints)
{
	bool isDiscontinuous[cloud->size()];
  int numClusters = 0;

  //for(int i = cloud->size() ; i >= 0 ; i--)
    //if(pcl::euclideanDistance((*cloud)[i],PointXYZ(0,0,0))<1.0)
      //cloud->erase(cloud->begin()+i);      	

	for(int i = 0 ; i < cloud->size() ; ++i)
	{
	    if(pcl::euclideanDistance((*cloud)[i],(*cloud)[(i+1)%cloud->size()])>threshold)
	    {
	      isDiscontinuous[(i+1)%cloud->size()] = true;
	      numClusters++;
	    }else{
	      isDiscontinuous[(i+1)%cloud->size()] = false;
	    }
  	}

  int clustersArray[numClusters];
  int n = 0;

	pcl::PointCloud<pcl::PointXYZ> cloud_cluster;
  	
  	for(int i = 0; i < cloud->size(); ++i)
    	if(isDiscontinuous[i])
      		clustersArray[n++]=i;

  	for(int i=0; i<numClusters; ++i)
  	{
  		int pointsPerCluster = (clustersArray[(i+1)%numClusters]-clustersArray[i]+cloud->size())%cloud->size();//numero de puntos de cada cluster
   		
   		if (pointsPerCluster<minClusterPoints || pointsPerCluster>maxClusterPoints)
   			continue;
    
    	cloud_cluster.clear();
    	
    	for(int j=clustersArray[i] ; j!=clustersArray[(i+1)%numClusters] ; j=(j+1)%cloud->size())
    		cloud_cluster.push_back((*cloud)[j]);

    	clusters.push_back(cloud_cluster);
	    /*std::stringstream ss2;
	    ss2 << "velodyne_rings_clusters/object" << frame<<"_"<<ring<<"_"<<i<<".pcd";
	    pcl::io::savePCDFile<pcl::PointXYZ>(ss2.str (), cloud_cluster);*/	
    }
}

/*Extract corner in a layer*/
void getCorners(PointCloud<PointXYZ> in, PointCloud<PointXYZ>& corners)
{
  
  PointCloud<PointXYZ>::Ptr tmp (new PointCloud<PointXYZ>);
  copyPointCloud<PointXYZ>(in, *tmp);
  vector<LinePair> linePairVector;
  vector< vector<int> > lineInliersVector;

  //Primero hallar todas las rectas posibles
  //while(tmp->points.size()>5)
  while(tmp->points.size()>10)
  {
    vector<int> lineInliers;
    SampleConsensusModelLine<PointXYZ>::Ptr modelLine (new SampleConsensusModelLine<PointXYZ> (tmp));
    RandomSampleConsensus<PointXYZ> ransac (modelLine);
    //ransac.setDistanceThreshold (.015);
    ransac.setDistanceThreshold (.02);
    ransac.computeModel();
    ransac.getInliers(lineInliers);

    vector<int> samples;
    samples.push_back(lineInliers[0]);
    samples.push_back(lineInliers[lineInliers.size()-1]);

    Eigen::VectorXf line(6);

    modelLine->computeModelCoefficients (samples, line);
    modelLine->optimizeModelCoefficients (lineInliers, line, line);


    PointCloud<PointXYZ> lineCloud;
    copyPointCloud<PointXYZ>(*tmp, lineInliers, lineCloud);

    LinePair linePair;
    linePair.line = line;
    linePair.lineCloud = lineCloud;

    //linesVector.push_back(line);
    linePairVector.push_back(linePair);

    PointIndices::Ptr lineIndices(new PointIndices);
    lineIndices->indices = lineInliers;
    ExtractIndices<PointXYZ> ex;
    ex.setInputCloud (tmp);
    ex.setIndices (lineIndices);
    ex.setNegative (true);
    ex.filter (*tmp);
  }

  int i1, i2;
  Eigen::VectorXf line[2];
  PointXYZ cornerFirst(numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max());

  //recorro todas las rectas para ver cuales se intersectan
  for(vector<LinePair>::iterator i=linePairVector.begin(); i!=linePairVector.end(); ++i)
  {
    for(vector<LinePair>::iterator j=linePairVector.begin(); j!=linePairVector.end(); ++j)
    {
      if( i1!=i2 )
      {

        Eigen::Vector3f v1((*i).line[3], (*i).line[4], (*i).line[5]);
        Eigen::Vector3f v2((*j).line[3], (*j).line[4], (*j).line[5]);
        float ang = acos(v1.dot(v2))*180/M_PI;

        //cout<<"angulo: "<<ang<<endl;
        //if(ang>80 && ang<100)
        if(ang>85 && ang<100)
        {
          line[0] = (*i).line;
          line[1] = (*j).line;

          Matrix2f A;
          Vector2f b;

          A << line[0][3], line[1][3]*(-1), line[0][4], line[1][4]*(-1);
          b << line[1][0] - line[0][0], line[1][1] - line[0][1];

          Vector2f x = A.colPivHouseholderQr().solve(b); //Solve the intersection between 2 lines with Eigen library

          float x1 = line[0][0] + line[0][3] * x[0];
          float y1 = line[0][1] + line[0][4] * x[0];
          float z1 = line[0][2] + line[0][5] * x[0];
     
          PointXYZ cornerCurr(x1, y1, z1);
          
          //cout<<euclideanDistance( corner, (*i).lineCloud.back() )<<endl;
          //cout<<euclideanDistance( corner, (*j).lineCloud.front() )<<endl;
          
          if(euclideanDistance( cornerCurr, (*i).lineCloud.back() ) <0.05 && euclideanDistance( cornerCurr, (*j).lineCloud.front()) < 0.05 && euclideanDistance( cornerFirst, cornerCurr) > 0.25)
          {
            corners.push_back(cornerCurr);
            //corners.push_back(PointXYZ(cornerCurr.x, cornerCurr.y, 0.0));
          }
          cornerFirst = cornerCurr;
        }
      }
      i2++;
    }
    i1++;
    i2 = 0;
  }
}

//Usar modulo de clustering euclideano
void separateFeatures(PointCloud<PointXYZ>::Ptr cornersAllLasers, PointCloud<PointXYZ> &cornersCloud, PointCloud<InterestPoint> &featureCloud, float separationThreshold)
{
  float featNumber = -1;

  pcl::search::KdTree<PointXYZ>::Ptr tree (new pcl::search::KdTree<PointXYZ>);
  tree->setInputCloud (cornersAllLasers);

  std::vector<pcl::PointIndices> cluster_indices;
  EuclideanClusterExtraction<PointXYZ> ec;
  ec.setClusterTolerance (separationThreshold); // 0.02 = 2cm
  ec.setMinClusterSize (1);
  ec.setMaxClusterSize (32);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cornersAllLasers);
  ec.extract (cluster_indices);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    float sumX = 0.0;
    float sumY = 0.0;

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      sumX += cornersAllLasers->points[*pit].x;
      sumY += cornersAllLasers->points[*pit].y;
    }

    featNumber++;
    InterestPoint feature;
    feature.x = sumX/it->indices.size();
    feature.y = sumY/it->indices.size();
    feature.z = 0;
    feature.strength = featNumber;
    featureCloud.push_back(feature);

    cornersCloud.push_back(PointXYZ(feature.x, feature.y, feature.z));
  }
}


void extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector< pcl::PointCloud<PointXYZ> > cloudVector)
{
  clock_t begin1 = clock();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cornersAllLasers(new pcl::PointCloud<pcl::PointXYZ>);
  
  cornersMerged->clear();

  if(isOdometryEnabled)
  {
  	cornersAllLasers->header.seq=cloud->header.seq;
  	cornersAllLasers->header.stamp=cloud->header.stamp;
  	cornersAllLasers->header.frame_id="/camera";

  	cornersMerged->header.seq=cloud->header.seq;
  	cornersMerged->header.stamp=cloud->header.stamp;
  	cornersMerged->header.frame_id="/camera";

  	cornerFeatures->header.seq=cloud->header.seq;
  	cornerFeatures->header.stamp=cloud->header.stamp;
  	cornerFeatures->header.frame_id="/camera_init";
  }
  else
  {
  	cornersAllLasers->header =cloud->header;
  	cornersMerged->header = cloud->header;
  }


  for(int i=0; i<12; ++i)
  {
	  featuresCloudPerLaserTotal[i].header.seq=cloud->header.seq;
  	  featuresCloudPerLaserTotal[i].header.stamp=cloud->header.stamp;
  	  featuresCloudPerLaserTotal[i].header.frame_id="/camera_init";
  	  //featuresCloudPerLaserTotal[i].header.frame_id="/velodyne";

  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTmp (new pcl::PointCloud<pcl::PointXYZ>);

  int i=0, j=0, k=0;

  pcl::PointCloud<pcl::PointXYZ> featuresCloudPerLaser[12];
  

  featureCloud.header.seq=cloud->header.seq;
  featureCloud.header.stamp=cloud->header.stamp;
  featureCloud.header.frame_id="/camera_init";

  //cout<<"Size: "<<cloudVector.size();

  //evaluate per layer
  for(vector< PointCloud<PointXYZ> >::iterator it = cloudVector.begin()+4 ; it != cloudVector.end(); ++it)
  //for(i = 0 ; i < 16; ++i)
  {	
  		*cloudTmp = *it;
  		//*cloudTmp[i] = cloudVector[i];
		
	    //std::stringstream ss1;
	    //ss1 << "velodyne_rings_clusters/objects" << frame<<"_"<<ring<<".pcd";
	    //pcl::io::savePCDFile<pcl::PointXYZ>(ss1.str (), *cloudTmp[i]);

	    vector< PointCloud<PointXYZ> > clusters;
	    clusterize(cloudTmp, clusters, CLUSTER_DISTANCE, MIN_CLUSTER_POINTS, MAX_CLUSTER_POINTS);
	    
	    //clusterize(cloudTmp[i], clusters[i], CLUSTER_DISTANCE);
	    j = 0;

	    //clusters
	    for (vector< PointCloud<PointXYZ> >::iterator it2 = clusters.begin() ; it2 != clusters.end(); ++it2)
	    //for (vector< PointCloud<PointXYZ> >::iterator it2 = clusters[i].begin() ; it2 != clusters[i].end(); ++it2)
	    {
	    	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
			  //*cloud_cluster = *it2;
        	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_features (new pcl::PointCloud<pcl::PointXYZ>);
        	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_global_features (new pcl::PointCloud<pcl::PointXYZ>);
        	getCorners(*it2, *cloud_features);
        
        	*cornersAllLasers += *cloud_features;
        	//cornersAllLasers += *cloud_global_features;

			    //*globalFeatures += *cloud_global_features;

			featuresCloudPerLaser[i] += *cloud_features;

	    	j++;
	    }

      //*cornersAnt[i] = featuresCloudPerLaser[i];
	    i++;
  }
  //}


  cout<<"Size: "<<i<<endl;;
  //separateFeatures(cornersAllLasers, *cornersMerged, featureCloud, 0.15);


  if(isOdometryEnabled)
  {
    pcl::transformPointCloud (*cornersMerged, *cornersMerged, transGlobal);
    pcl::transformPointCloud (featureCloud, featureCloud, transGlobal);
  }

  *cornerFeatures += cornersGlobal; //Se guardará al final para ver los features acumulados
  //*cornerFeatures += *cornersMerged;

  /*pcl::PCLPointCloud2 cloud2;
  cloud2.header=cloud->header;
  toPCLPointCloud2(*cornersAllLasers, cloud2);
  pub1.publish (cloud2);*/

  PCLPointCloud2 cloudMsg1;
  cloudMsg1.header=cloud->header;
  //toPCLPointCloud2(*cornersMerged, cloudMsg1);
  toPCLPointCloud2(*cornersAllLasers, cloudMsg1);
  pub1.publish (cloudMsg1);

  /*pcl::PCLPointCloud2 cloud3;
  cloud3.header=cloud->header;
  toPCLPointCloud2(featuresCloudPerLaserTotal[0], cloud3);
  pub2.publish (cloud3);*/

  pcl::PCLPointCloud2 cloudMsg2;
  cloudMsg2.header=cloud->header;
  toPCLPointCloud2(*cornerFeatures, cloudMsg2);
  pub2.publish (cloudMsg2);

  /*pcl::PCLPointCloud2 cloud2;
  cloud2.header=cloud->header;

  toPCLPointCloud2(cornersAllLasers, cloud2);
  pub1.publish (cloud2);

  sensor_msgs::PointCloud2 features_msg;
  pcl::toROSMsg<pcl::InterestPoint>(features,features_msg);
  pub2.publish (features_msg);*/

  clock_t end1 = clock();  
  double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
  cout<<"************************************"<<endl<<endl;
  cout<<"TIEMPO: "<<elapsed_secs1<<endl;
  cout<<"************************************"<<endl<<endl;
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "corner_extraction_test");
  ros::NodeHandle nh;

  omp_set_num_threads(4);

  //Transformacion de LOAM
  transGlobal = Eigen::Affine3f::Identity();
  transGlobal.translation() << 0.0, 0.0, 0.0;
  transGlobal.rotate (Eigen::AngleAxisf (-M_PI/2.0, Eigen::Vector3f::UnitX())); //data obtained from tf static_transform node
  transGlobal.rotate (Eigen::AngleAxisf (-M_PI/2.0, Eigen::Vector3f::UnitZ())); //data obtained from tf static_transform node

  for(int i=0; i<12; ++i)
    cornersAnt[i] = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub1 = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, cloud_callback);
  ros::Subscriber sub2 = nh.subscribe<nav_msgs::Odometry> ("/integrated_to_init", 1, odom_callback);
  //ros::Subscriber sub2 = nh.subscribe<nav_msgs::Odometry> ("/aft_mapped_to_init", 1, odom_callback);
  //ros::Subscriber sub2 = nh.subscribe<nav_msgs::Odometry> ("/encoder", 1, odom_callback);
  ros::Subscriber sub3 = nh.subscribe<sensor_msgs::PointCloud2> ("/laser_cloud_sharp", 1, cloud_callback2);
  
  pub1 = nh.advertise<sensor_msgs::PointCloud2> ("/features1", 1);
  pub2 = nh.advertise<sensor_msgs::PointCloud2> ("/features2", 1);
  ros::spin ();

  //pcl::io::savePCDFile<pcl::PointXYZ>("globalCloud.pcd", *globalCloud);
  pcl::io::savePCDFile<pcl::PointXYZ>("/home/claydergc/CornerFeatures/cornerFeatures.pcd", *cornerFeatures);
  pcl::io::savePCDFile<pcl::PointXYZ>("/home/claydergc/CornerFeatures/loamSharp.pcd", *loamSharp);
}