#include <opencv2/opencv.hpp> 
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

#define NBCAMERA 5

int main (int argc,char **argv)
{
    map<int,VideoCapture> v;

    for (int i = 0; i<NBCAMERA;i++)
    {
		VideoCapture vid(i);
		if (!vid.isOpened())
		{
			cout << "La caméra " << i << "ne peut être ouverte."<< endl;
		}
		else
			v.insert(make_pair(i, vid));
    }

    vector<Mat> frame(v.size());
    char c=0;
    for (;c!=27;)
    {
        map<int, VideoCapture>::iterator ite = v.begin();
        for (size_t i = 0; i<v.size() && c!=27;i++,ite++)
        {
            if (v[static_cast<int>(i)].isOpened())
            {

                ite->second >> frame[static_cast<int>(i)];
                imshow(format("Main WebCam %d", i), frame[static_cast<int>(i)]);
                c=waitKey(1);
            }
        }
    }
    map<int, VideoCapture>::iterator ite = v.begin();
	for (size_t i = 0; i<v.size(); i++,ite++)
		ite->second.release();
return 0;
}

