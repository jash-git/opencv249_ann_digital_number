#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <stdio.h>
using namespace cv;
using namespace std;
#define TRAIN false    //是否进行训练,true表示重新训练，false表示读取xml文件中的ann模型

int main()
{
	//	const string fileform = "*.png";
	//  const string perfileReadPath ;

	const int sample_mun_perclass = 20;//训练字符每类数量
	const int class_mun = 10;//训练字符类数

	const int image_cols = 8;
	const int image_rows = 16;
	const string fileForm = ".txt";
	const string fileReadName = "D:/Opencv2.4.9/VS2013Project/myAnn/myAnn/myCharPic/";
	string fileReadPath;
	string txtName;
	char tmp[10];
	string ImgName;
	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//每一行一个训练样本
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//训练样本标签
	CvANN_MLP bp;
	//ifstream fin(txtName);//正样本图片的文件名列表

	if (TRAIN)
	{
		for (int i = 0; i<class_mun; ++i)//不同类
		{
			//读取每个类文件夹下所有图像
			sprintf(tmp, "%d", i);
			cout << "文件夹" << i << endl;
			txtName = tmp + fileForm;
			ifstream fin(txtName);//正样本图片的文件名列表
			for (int j = 1; j <= sample_mun_perclass; j++)
			{
				getline(fin, ImgName);
				cout << "处理：" << ImgName << endl;
				fileReadPath = fileReadName + tmp + "/" + ImgName;
				Mat srcImage = imread(fileReadPath, 0);//读取图片
				Mat resizeImage;
				Mat trainImage;
				Mat result;

				resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
				threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

				for (int k = 0; k<image_rows*image_cols; ++k)
				{
					trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];
					//trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
					//cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
				}

			} //如果设置读入的图片数量，则以设置的为准，如果图片不够，则读取文件夹下所有图片

		}

		// Set up training data Mat
		Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
		cout << "trainingDataMat——OK！" << endl;

		// Set up label data 
		for (int i = 0; i <= class_mun - 1; ++i)
		{
			for (int j = 0; j <= sample_mun_perclass - 1; ++j)
			{
				for (int k = 0; k<class_mun; ++k)
				{
					if (k == i)
						labels[i*sample_mun_perclass + j][k] = 1;
					else labels[i*sample_mun_perclass + j][k] = 0;
				}
			}
		}
		Mat labelsMat(class_mun*sample_mun_perclass, class_mun, CV_32FC1, labels);
		cout << "labelsMat:" << endl;
		cout << labelsMat << endl;
		cout << "labelsMat——OK！" << endl;

		//训练代码

		cout << "training start...." << endl;

		// Set up BPNetwork's parameters
		CvANN_MLP_TrainParams params;
		params.train_method = CvANN_MLP_TrainParams::BACKPROP;
		params.bp_dw_scale = 0.001;
		params.bp_moment_scale = 0.1;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001);  //设置结束条件
		//params.train_method=CvANN_MLP_TrainParams::RPROP;
		//params.rp_dw0 = 0.1;
		//params.rp_dw_plus = 1.2;
		//params.rp_dw_minus = 0.5;
		//params.rp_dw_min = FLT_EPSILON;
		//params.rp_dw_max = 50.;

		//Setup the BPNetwork
		Mat layerSizes = (Mat_<int>(1, 5) << image_rows*image_cols, 128, 128, 128, class_mun);
		bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);//CvANN_MLP::SIGMOID_SYM
		//CvANN_MLP::GAUSSIAN
		//CvANN_MLP::IDENTITY
		cout << "training...." << endl;
		bp.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

		bp.save("../bpcharModel.xml"); //save classifier
		cout << "training finish...bpModel1.xml saved " << endl;
	}
	else //若TRAIN为false，从XML文件读取训练好的分类器
	{
		bp.load("../bpcharModel.xml");//从XML文件读取训练好的SVM模型
	}

	//测试神经网络
	cout << "测试：" << endl;
	Mat test_image = imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_temp;
	resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
	threshold(test_temp, test_temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Mat_<float>sampleMat(1, image_rows*image_cols);
	for (int i = 0; i<image_rows*image_cols; ++i)
	{
		sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 8, i % 8);
	}

	Mat responseMat;
	bp.predict(sampleMat, responseMat);
	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);
	cout << "识别结果：" << maxLoc.x << "	相似度:" << maxVal * 100 << "%" << endl;
	imshow("test_image", test_image);
	waitKey(0);

	return 0;
}