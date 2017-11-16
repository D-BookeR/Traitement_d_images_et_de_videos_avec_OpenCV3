#include <opencv2/opencv.hpp> 
#include <opencv2/text.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace text;


const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |texte7.png     | nom de l'image }";

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String nomFichierImage = parser.get<String>(0);

    Mat mOriginal = imread(nomFichierImage, IMREAD_UNCHANGED);
    Mat mOriginalRGB;
    if (mOriginal.empty())
    {
        cout << "Image vide!\n";
        return -1;
    }
    vector<Mat> tabCanaux;
    cvtColor(mOriginal, mOriginalRGB,CV_BGR2RGB);
    computeNMChannels(mOriginalRGB,tabCanaux,ERFILTER_NM_RGBLGrad);
    int cn = (int)tabCanaux.size();
    for (int c = 0; c < cn - 1; c++)
        tabCanaux.push_back(255 - tabCanaux[c]);

    Ptr<ERFilter> filtreRegion1 = createERFilterNM1(loadClassifierNM1("g:/lib/opencv_contrib/modules/text/samples/trained_classifierNM1.xml"), 16, 0.00005f, 0.13f, 0.2f, true, 0.1f);
    Ptr<ERFilter> filtreRegion2 = createERFilterNM2(loadClassifierNM2("g:/lib/opencv_contrib/modules/text/samples/trained_classifierNM2.xml"), 0.5);

    vector<vector<ERStat> > regions(tabCanaux.size());
    Mat zoneCaracteres=mOriginal.clone();

    vector< vector<Vec2i> > indexZone;
    vector<Rect> zoneTexte;
    for (int i = 0; i<(int)tabCanaux.size(); i++)
    {
        cout << "Traitement canal "<<i<<endl;
        filtreRegion1->run(tabCanaux[i], regions[i]);
        for (int j = 0; j<regions[i].size(); j++)
            rectangle(zoneCaracteres, regions[i][j].rect, Scalar(0, 0, 255),3);
        filtreRegion2->run(tabCanaux[i], regions[i]);
        for (int j=0;j<regions[i].size();j++)
            rectangle(zoneCaracteres, regions[i][j].rect,Scalar(255,0,0),1);
    }

    imshow("Zone de caracteres", zoneCaracteres);
    waitKey(10);
    erGrouping(mOriginal, tabCanaux, regions, indexZone, zoneTexte, 0.5);
//    erGrouping(mOriginal, tabCanaux, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY, "g:/lib/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml", 0.5);

    for (int i = 0; i<zoneTexte.size(); i++)
        rectangle(mOriginal, zoneTexte[i],Scalar::all(255));
    imshow("Zone de texte", mOriginal);
    waitKey();
    return 0;
}
