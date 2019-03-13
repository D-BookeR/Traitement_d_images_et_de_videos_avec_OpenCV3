#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | chemin complet de l'image}"
;

cv::String nomFenetre = "Classes";
cv::String nomOriginal = "image Originale";

struct ParamKMean {
    int couleur;
    Mat img;
    Mat imgGris;
    int nbClasses;
    int essai;
    TermCriteria critere;
    Mat nuageCouleur;
    Mat nuageGris;
    Mat lutRND;
};

void Classification(ParamKMean *t);
void EffaceFenetre(String , Mat);
void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r);

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String nomFic = parser.get<String>(0);
    if (nomFic.length() == 0)
    {
        parser.printMessage();
        return 0;
    }
    namedWindow(nomFenetre);
    ParamKMean t;
    t.couleur=0;
    t.critere = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1);
    t.nbClasses = 2;
    t.essai=1;
    AjouteGlissiere("Classes", nomFenetre, 2, 64, t.nbClasses, &t.nbClasses, NULL, NULL);
    AjouteGlissiere("Iteration", nomFenetre, 1, 20, t.critere.maxCount, &t.critere.maxCount, NULL, NULL);
    AjouteGlissiere("Essai", nomFenetre, 1, 20, t.essai, &t.essai, NULL, NULL);
    AjouteGlissiere("Couleur", nomFenetre, 0, 1, t.couleur, &t.couleur, NULL, NULL);
    t.img = imread(nomFic, IMREAD_COLOR);
    if (t.img.empty())
    {
        cout << "Fichier vide ou inexistant. Vérifiez le nom du fichier.\n";
        return 0;
    }
    cvtColor(t.img,t.imgGris,COLOR_BGR2GRAY);
    imshow(nomOriginal, t.imgGris);
    Mat srcFGris,srcFcouleur;
    t.img.convertTo(srcFcouleur, CV_32FC3);
    t.nuageCouleur = srcFcouleur.reshape(0,t.img.cols*t.img.rows);
    t.imgGris.convertTo(srcFGris, CV_32FC1);
    t.nuageGris = srcFGris.reshape(0, t.img.cols*t.img.rows);
    if (t.lutRND.empty())
    {
        RNG r;
        t.lutRND = Mat(256, 1, CV_8UC3);
        r.fill(t.lutRND, RNG::UNIFORM, 0, 256);
    }
    char code=0;
    while (code != 27)
    {
        code=waitKey();
        if (code == 'g')
        {
            EffaceFenetre(nomFenetre,t.img);
            Classification(&t);
        }
    }
    return 0;
}

void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r)
{
    createTrackbar(nomGlissiere, nomFenetre, valGlissiere, 1, f, r);
    setTrackbarMin(nomGlissiere, nomFenetre, minGlissiere);
    setTrackbarMax(nomGlissiere, nomFenetre, maxGlissiere);
    setTrackbarPos(nomGlissiere, nomFenetre, valeurDefaut);
}

void EffaceFenetre(String nomFenetre, Mat x)
{
    Mat y=Mat::zeros(x.size(),CV_8UC1);
    putText(y,"Classification en cours!",Point(10,x.rows/2), FONT_HERSHEY_SIMPLEX,1,Scalar(255));
    imshow(nomFenetre,y);
    waitKey(1);
}

void Classification(ParamKMean *t)
{
    Mat labels;
    Mat dst, mask;
    if (t->couleur)
    {
        kmeans(t->nuageCouleur, t->nbClasses, labels, t->critere, t->essai, KMEANS_PP_CENTERS);
        imshow(nomOriginal, t->img);
    }
    else
    {
        kmeans(t->nuageGris, t->nbClasses, labels, t->critere, t->essai, KMEANS_PP_CENTERS);
        imshow(nomOriginal, t->imgGris);
    }
    Mat result;
    labels.convertTo(result,CV_8UC1);
    result =  result.reshape(0, t->img.rows);
    vector<Mat> xx = { result,result,result };
    Mat cc;
    merge(xx, cc);
    LUT(cc, t->lutRND, cc);
    imshow(nomFenetre, cc);
}


