#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib> //srand() ,rand() ,system()
#include <ctime>   //time()

#include <sys/timeb.h>
#if defined(WIN32)
    #define  TIMEB    _timeb
    #define  ftime    _ftime
    typedef __int64 TIME_T;
#else
    #define TIMEB timeb
    typedef long long TIME_T;
#endif

using namespace cv;
using namespace std;

void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}
int Rand_M2N(int intstart,int intrange)
{
    srand(time(NULL));
    return intstart+(rand()%intrange);
}
int main()
{
    const int image_cols = 8;
	const int image_rows = 16;
    char ad[128]={0};
    Mat traindata ,trainlabel;
    //读取训练数据 4000张
    for (int i = 0; i < 10; i++)
    {
        for (int j =0;j<500;j++)
        {
             sprintf(ad, "data\\%d\\%d.jpg",i,j);
             Mat srcImage = imread(ad,0);
             Mat resizeImage;
             Mat trainImage;
             resize(srcImage, resizeImage, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
			 threshold(resizeImage, trainImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

             /*
                Mat::reshape( ) 
                只是在逻辑上改变矩阵的行列数或者通道数，没有任何的数据的复制，也不会增减任何数据，因此这是一个O(1)的操作，它要求矩阵是连续的。
                C++: Mat Mat::reshape(int cn, int rows=0 const)

                cn：目标通道数，如果是0则保持和原通道数一致；

                rows：目标行数，同上是0则保持不变；

                改变后的矩阵要满足 rows*cols*channels  跟原数组相等，所以如果原来矩阵是单通道3*3的，调用Reshape(0,2)是会报错的，因为3*3*1不能被2*1整除。
             */
             trainImage = trainImage.reshape(1,1);
             traindata.push_back(trainImage);//将元素添加到矩阵的底部。（在为mat增加一行的时候，用到push_back）

             float responses[10]={0,0,0,0,0,0,0,0,0,0};
             responses[i]=1;
             Mat responsesMat(1, 10, CV_32FC1, responses);
             trainlabel.push_back(responsesMat);//将元素添加到矩阵的底部。（在为mat增加一行的时候，用到push_back）
        }
    }

    traindata.convertTo(traindata,CV_32F);
    trainlabel.convertTo(trainlabel,CV_32F);

	CvANN_MLP_TrainParams params(
        cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100000000, 0.00001),    //终止条件
        CvANN_MLP_TrainParams::BACKPROP,    // BACKPROP算法
        0.1, 0.1);    //激活函数的两个参数 [一個是權值更新率bp_dw_scale和權值更新衝量bp_moment_scale。這兩個量一般情況設定為0.1就行了；太小了網路收斂速度會很慢，太大了可能會讓網路越過最小值點。]
    //4层MLP，输入层有400个神经元，隐藏层01有400个神经元，隐藏层02有400个神经元，输出层有10个神经元
	Mat layerSizes = (Mat_<int>(1,3) << 128, 128*2, 10);

	CvANN_MLP bp;    //实例化MLP
    //创建MLP模型，选用的激励函数为对称SIGMOID函数
    bool bnload=true;
    if(!bnload)
    {
        bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);
        bool check=bp.train(traindata, trainlabel, Mat(),Mat(), params);  //训练MLP模型
        if(check==true)
        {
            bp.save("ann_param");
            cout << "train ok\n";
        }
    }
    else
    {
        bp.load("ann_param");
    }


    /*
    RNG rng;
    int digital=rng.uniform(0, 9);
    int index=rng.uniform(400, 499);
    */
    int digital=Rand_M2N(0,(9-0));
    int index=Rand_M2N(0,(499-0));
    sprintf(ad, "data\\%d\\%d.jpg",digital,index);
    cout << "test image path : " << ad << endl;

    Mat testdata = imread(ad,0);
    Mat showdata=testdata.clone();
    resize(testdata, testdata, Size(image_cols, image_rows), (0, 0), (0, 0), CV_INTER_AREA);
    threshold(testdata, testdata, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


    testdata = testdata.reshape(1,1);
    testdata.convertTo(testdata,CV_32F);

    Mat p_output_01;
    bp.predict(testdata, p_output_01);
    cout << "output:" << endl << p_output_01 << endl << endl;

	Point maxLoc;
	double maxVal = 0;
	minMaxLoc(p_output_01, NULL, &maxVal, NULL, &maxLoc);
	cout << "ANS:" << maxLoc.x << "\t" << maxVal * 100 << "%" << endl;

    namedWindow(ad, CV_WINDOW_NORMAL);
    imshow(ad,showdata);

    waitKey(0);
    Pause();
    return 0;
}
