#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <stdio.h>
using namespace cv;
using namespace std;
#define TRAIN false    //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�annģ��

int main()
{
	//	const string fileform = "*.png";
	//  const string perfileReadPath ;

	const int sample_mun_perclass = 20;//ѵ���ַ�ÿ������
	const int class_mun = 10;//ѵ���ַ�����

	const int image_cols = 8;
	const int image_rows = 16;
	const string fileForm = ".txt";
	const string fileReadName = "D:/Opencv2.4.9/VS2013Project/myAnn/myAnn/myCharPic/";
	string fileReadPath;
	string txtName;
	char tmp[10];
	string ImgName;
	float trainingData[class_mun*sample_mun_perclass][image_rows*image_cols] = { { 0 } };//ÿһ��һ��ѵ������
	float labels[class_mun*sample_mun_perclass][class_mun] = { { 0 } };//ѵ��������ǩ
	CvANN_MLP bp;
	//ifstream fin(txtName);//������ͼƬ���ļ����б�

	if (TRAIN)
	{
		for (int i = 0; i<class_mun; ++i)//��ͬ��
		{
			//��ȡÿ�����ļ���������ͼ��
			sprintf(tmp, "%d", i);
			cout << "�ļ���" << i << endl;
			txtName = tmp + fileForm;
			ifstream fin(txtName);//������ͼƬ���ļ����б�
			for (int j = 1; j <= sample_mun_perclass; j++)
			{
				getline(fin, ImgName);
				cout << "����" << ImgName << endl;
				fileReadPath = fileReadName + tmp + "/" + ImgName;
				Mat srcImage = imread(fileReadPath, 0);//��ȡͼƬ
				Mat resizeImage;
				Mat trainImage;
				Mat result;

				resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
				threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

				for (int k = 0; k<image_rows*image_cols; ++k)
				{
					trainingData[i*sample_mun_perclass + (j - 1)][k] = (float)trainImage.data[k];
					//trainingData[i*sample_mun_perclass+(j-1)][k] = (float)trainImage.at<unsigned char>((int)k/8,(int)k%8);//(float)train_image.data[k];
					//cout<<trainingData[i*sample_mun_perclass+(j-1)][k] <<" "<< (float)trainImage.at<unsigned char>(k/8,k%8)<<endl;
				}

			} //������ö����ͼƬ�������������õ�Ϊ׼�����ͼƬ���������ȡ�ļ���������ͼƬ

		}

		// Set up training data Mat
		Mat trainingDataMat(class_mun*sample_mun_perclass, image_rows*image_cols, CV_32FC1, trainingData);
		cout << "trainingDataMat����OK��" << endl;

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
		cout << "labelsMat����OK��" << endl;

		//ѵ������

		cout << "training start...." << endl;

		// Set up BPNetwork's parameters
		CvANN_MLP_TrainParams params;
		params.train_method = CvANN_MLP_TrainParams::BACKPROP;
		params.bp_dw_scale = 0.001;
		params.bp_moment_scale = 0.1;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001);  //���ý�������
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
	else //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	{
		bp.load("../bpcharModel.xml");//��XML�ļ���ȡѵ���õ�SVMģ��
	}

	//����������
	cout << "���ԣ�" << endl;
	Mat test_image = imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat test_temp;
	resize(test_image, test_temp, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//ʹ�����ع�ϵ�ز�������ͼ����Сʱ�򣬸÷������Ա��Ⲩ�Ƴ���
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
	cout << "ʶ������" << maxLoc.x << "	���ƶ�:" << maxVal * 100 << "%" << endl;
	imshow("test_image", test_image);
	waitKey(0);

	return 0;
}