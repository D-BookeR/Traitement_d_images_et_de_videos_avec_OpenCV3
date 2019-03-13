#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_XIMGPROC
#include <opencv2/ximgproc/deriche_filter.hpp>
#endif

using namespace cv;
using namespace std;

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | chemin complet de l'image couleur (3 canaux)}";

enum TypeGradient{FILTRE_SOBEL=0,FILTRE_SCHARR,FILTRE_DERICHE};

struct ParamCanny {
    bool changementFiltre;
    TypeGradient typeGradient;
    string nomFiltre;
    int taille;
#ifdef HAVE_OPENCV_XIMGPROC
	int alphaDerive;
    int alphaMean;
#endif
	int seuilMin;
    int seuilMax;
    int maxGradient;
    Mat img;
    Mat gx,gy;
    Mat lutRND;
};


static void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r = NULL);
static void CalculGradient(ParamCanny *r);
static void ChoixSeuil(int x, void *r);
static void ChoixGradient(int x, void *r);
static void TailleSobel(int x, void *r);
#ifdef HAVE_OPENCV_XIMGPROC
static void DericheParametre(int x, void *r);
#endif

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
    String fenetreImage = "Image Originale";
    ParamCanny pgc;
    pgc.changementFiltre=false;
    pgc.typeGradient = FILTRE_SOBEL;
    pgc.seuilMin = 200;
    pgc.seuilMax = 500;
#ifdef HAVE_OPENCV_XIMGPROC
	pgc.alphaDerive = 100;
    pgc.alphaMean = 100;
#endif
	pgc.nomFiltre="Sobel";
    pgc.taille =1;
    pgc.maxGradient = 1000;
    pgc.img = imread(nomFic, IMREAD_COLOR);
	if (pgc.img.empty())
	{
		cout << "Image vide.";
		exit(1);
	}
    if (pgc.lutRND.empty())
    {
        RNG r;
        pgc.lutRND = Mat(65536, 1, CV_8UC3);
        r.fill(pgc.lutRND, RNG::UNIFORM, 0, 256);
    }
    imshow(fenetreImage,pgc.img);
    int valFiltre;
    AjouteGlissiere("Filtre", fenetreImage, 0, 2, pgc.typeGradient, &valFiltre, ChoixGradient, (void*)&pgc);
    AjouteGlissiere("Taille", fenetreImage, 0, 3, pgc.taille, &pgc.taille, TailleSobel, (void*)&pgc);
    AjouteGlissiere("Seuil Min", fenetreImage, 0, 2 * pgc.maxGradient, pgc.seuilMin, &pgc.seuilMin, ChoixSeuil, (void*)&pgc);
    AjouteGlissiere("Seuil Max", fenetreImage, 0, 2 * pgc.maxGradient, pgc.seuilMax, &pgc.seuilMax, ChoixSeuil, (void*)&pgc);

    int code=0;
    do
    {
        code =  waitKey(30)&0xFF;
        if (pgc.changementFiltre)
        {
            destroyWindow(fenetreImage);
            imshow(fenetreImage, pgc.img);
            AjouteGlissiere("Filtre", fenetreImage, 0, 2, pgc.typeGradient, &valFiltre, ChoixGradient, (void*)&pgc);
            AjouteGlissiere("Seuil Min", fenetreImage, 0, 2* pgc.maxGradient, pgc.seuilMin, &pgc.seuilMin, ChoixSeuil, (void*)&pgc);
            AjouteGlissiere("Seuil Max", fenetreImage, 0, 2 * pgc.maxGradient, pgc.seuilMax, &pgc.seuilMax, ChoixSeuil, (void*)&pgc);
#ifdef HAVE_OPENCV_XIMGPROC
            if (pgc.typeGradient == FILTRE_DERICHE)
            {
                AjouteGlissiere("Derive", fenetreImage, 0, 400, pgc.alphaDerive, &pgc.alphaDerive, DericheParametre, (void*)&pgc);
                AjouteGlissiere("Moyenne", fenetreImage, 0, 400, pgc.alphaMean, &pgc.alphaMean, DericheParametre, (void*)&pgc);
            }
#endif
            if (pgc.typeGradient == FILTRE_SOBEL)
            {
                AjouteGlissiere("Taille", fenetreImage, 0, 3, pgc.taille, &pgc.taille, TailleSobel, (void*)&pgc);
            }
            waitKey(20);
            pgc.changementFiltre = false;
        }
    }
    while (code!=27);
    return 0;
}

void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r)
{
    createTrackbar(nomGlissiere, nomFenetre, valGlissiere, 1, f, r);
    setTrackbarMin(nomGlissiere, nomFenetre, minGlissiere);
    setTrackbarMax(nomGlissiere, nomFenetre, maxGlissiere);
    setTrackbarPos(nomGlissiere, nomFenetre, valeurDefaut);
}

void ChoixGradient(int x, void *r)
{
    ParamCanny *pgc = (ParamCanny*)r;
    if (x != pgc->typeGradient || pgc->gx.empty())
    {
        pgc->typeGradient = (TypeGradient)x;
        CalculGradient(pgc);
        ChoixSeuil(-1, r);
        pgc->changementFiltre = true;
    }
}

void CalculGradient(ParamCanny *pgc)
{
    switch (pgc->typeGradient)
    {
    case FILTRE_SOBEL:
        Sobel(pgc->img, pgc->gx, CV_16S, 1, 0, 2 * pgc->taille + 1);
        Sobel(pgc->img, pgc->gy, CV_16S, 0, 1, 2 * pgc->taille + 1);
        pgc->nomFiltre = "Sobel";
        break;
    case FILTRE_SCHARR:
        Scharr(pgc->img, pgc->gx, CV_16S, 1, 0);
        Scharr(pgc->img, pgc->gy, CV_16S, 0, 1);
        pgc->nomFiltre = "Scharr";
        break;
#ifdef HAVE_OPENCV_XIMGPROC
    case FILTRE_DERICHE:
        cv::ximgproc::GradientDericheX(pgc->img, pgc->gx, pgc->alphaDerive / 100.0, pgc->alphaMean / 100.0);
        cv::ximgproc::GradientDericheY(pgc->img, pgc->gy, pgc->alphaDerive / 100.0, pgc->alphaMean / 100.0);
        pgc->nomFiltre = "Deriche";
        break;
#endif
    }
    Mat extX = Mat::zeros(1, 12, CV_64FC1);
    minMaxIdx(pgc->gx, extX.ptr<double>(0), extX.ptr<double>(0)+3);
    minMaxIdx(pgc->gy, extX.ptr<double>(0)+6, extX.ptr<double>(0)+9);
    extX = abs(extX);
    double maxG;
    minMaxIdx(extX, NULL, &maxG);
    pgc->gx.convertTo(pgc->gx, CV_16S, pgc->maxGradient / maxG);
    pgc->gy.convertTo(pgc->gy, CV_16S, pgc->maxGradient / maxG);
}


void ChoixSeuil(int x, void *r)
{
    ParamCanny *pgc = (ParamCanny*)r;
    Mat front;
    if (pgc->gx.empty())
        CalculGradient(pgc);
    Canny(pgc->gx, pgc->gy, front, static_cast<double>(pgc->seuilMin), static_cast<double>(pgc->seuilMax));
    imshow(format("Canny %s", pgc->nomFiltre.c_str()), front);
    vector<vector<Point> > contours;
    Mat cmpConnex(front.size(),CV_8UC3,Scalar(0));
    findContours(front,contours, RETR_LIST, CHAIN_APPROX_NONE);
    for (int i = 0; i < contours.size(); i++)
    {
        drawContours(cmpConnex,contours,i,Scalar(pgc->lutRND.at<Vec3b>(i,0)), -1);
    }
    imshow("Remplissage des contours",cmpConnex);
}

void TailleSobel(int x, void *r)
{
    ParamCanny *pgc = (ParamCanny*)r;
    CalculGradient(pgc);
    ChoixSeuil(-1, r);
}

#ifdef HAVE_OPENCV_XIMGPROC
void DericheParametre(int x, void *r)
{
    ParamCanny *pgc = (ParamCanny*)r;
    CalculGradient(pgc);
    ChoixSeuil(-1, r);
}
#endif

