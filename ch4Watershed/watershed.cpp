#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

static void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r = NULL);

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | chemin complet de l'image couleur (3 canaux)}";

#define MODE_AFFICHAGE_SEED 0
#define MODE_AFFICHAGE_TRANSPARENCE 1

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
    Mat imgOriginal = imread(nomFic,IMREAD_COLOR);

    if (imgOriginal.empty())
    {
        cout << "Fichier vide ou inexistant. Vérifiez le nom du fichier.\n";
        return 0;
    }
    Mat lutRND;
    if (lutRND.empty())
    {
        RNG r;
        lutRND = Mat(256, 1, CV_8UC3);
        r.fill(lutRND, RNG::UNIFORM, 0, 256);
    }
    String nomFenetre="Graine";
    namedWindow(nomFenetre);
    int seuilGr=20;
    int seuilGrClone=seuilGr;
    AjouteGlissiere("Seuil grad.",nomFenetre,0, 255, seuilGrClone,&seuilGr,NULL,NULL);
    Mat gx,gy,g;
    
    GaussianBlur(imgOriginal, imgOriginal, Size(5, 5),1);
    Sobel(imgOriginal,gx, CV_32F, 1, 0,3);
    Sobel(imgOriginal,gy, CV_32F, 0, 1,3);
    magnitude(gx,gy,g);
    normalize(g,g,256,0,NORM_INF);
    g.convertTo(g,CV_8U);
    Mat gMax;
    cvtColor(g, gMax, CV_BGR2GRAY);
    Mat pointsBas =gMax<seuilGrClone;
    Mat eltStruct=getStructuringElement(MORPH_RECT,Size(3,3));
    char modeAffichage= MODE_AFFICHAGE_SEED;
    int code=0;
    Mat cc;
    do
    {
        code=waitKey(30)&0xFF;
        if (seuilGr != seuilGrClone)
        {
            pointsBas = gMax<seuilGr;
            seuilGrClone= seuilGr;
        }
        switch (code) {
        case 'e':
            erode(pointsBas, pointsBas, eltStruct);
            break;
        case 'd':
            dilate(pointsBas, pointsBas, eltStruct);
            break;
        case 'f':
            morphologyEx(pointsBas, pointsBas, MORPH_CLOSE, eltStruct);
            break;
        case 'o':
            morphologyEx(pointsBas, pointsBas, MORPH_OPEN, eltStruct);
            break;
        case 'r':
            pointsBas = gMax<seuilGr;
            break;
        case 't':
            modeAffichage = MODE_AFFICHAGE_TRANSPARENCE;
            break;
        case 's':
            modeAffichage = MODE_AFFICHAGE_SEED;
            break;
        case 'w':
            Mat res;
            int nb=connectedComponents(pointsBas, cc, 8, CV_32S);
            watershed(g, cc);
            cc.convertTo(res, CV_8UC1);
            cvtColor(res,res, COLOR_GRAY2BGR);
            LUT(res, lutRND, res);
            imshow("res", res);
            break;
        }
        if (modeAffichage == MODE_AFFICHAGE_TRANSPARENCE&& !cc.empty())
        {
            Mat imgAffi=Mat::zeros(imgOriginal.size(),CV_8UC3);
            imgOriginal.copyTo(imgAffi,cc!=-1);
            imshow(nomFenetre, imgAffi);
        }
        else if (modeAffichage== MODE_AFFICHAGE_SEED)
        {
            imshow(nomFenetre,  pointsBas);
        }
    }
    while(code!=27);
    return 0;
}

void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r)
{
    createTrackbar(nomGlissiere, nomFenetre, valGlissiere, 1, f, r);
    setTrackbarMin(nomGlissiere, nomFenetre, minGlissiere);
    setTrackbarMax(nomGlissiere, nomFenetre, maxGlissiere);
    setTrackbarPos(nomGlissiere, nomFenetre, valeurDefaut);
}

