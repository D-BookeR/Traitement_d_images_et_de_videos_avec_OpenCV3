#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | chemin complet de l'image couleur (3 canaux)}";

static void SeuilImageNB(int x, void *r);
static void SeuilImageNBAdaptatif(int x, void *r);
static void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int,void *), void *r=NULL);

cv::String nomFenetreNB = "Image Seuillée NB";
cv::String nomFenetreNBAda = "Image Seuillée (ada.)NB";
cv::String nomFenetreRGB = "Image Seuillée RGB";

struct ParamBinarisation {
	bool seuillageAdaptatif;
    int valSeuil;
    int valTypeSeuil;
    int adaptiveMethod;
    int valTypeSeuilAda;
    int blockSize;
    Mat mNB;
    Mat mBGR;
    Mat lutRND;
};


int main(int argc, char* argv[])
{
    cout<<getNumThreads();
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
    ParamBinarisation pBin;
 
    pBin.mBGR = imread(nomFic,IMREAD_COLOR);
    if (pBin.mBGR.empty())
    {
        cout << "Fichier vide ou inexistant. Vérifiez le nom du fichier.\n";
        return 0;
    }

    cvtColor(pBin.mBGR, pBin.mNB,CV_BGR2GRAY);
    Mat dst;
	pBin.seuillageAdaptatif = false;
    pBin.valSeuil = saturate_cast<int>(threshold(pBin.mNB, dst, 0, 255, THRESH_BINARY | THRESH_OTSU));
    pBin.valTypeSeuil = THRESH_BINARY;
    pBin.adaptiveMethod = ADAPTIVE_THRESH_MEAN_C;
    pBin.valTypeSeuilAda = THRESH_BINARY;
    pBin.blockSize=31;
    namedWindow(nomFenetreNB);
    AjouteGlissiere("Seuil", nomFenetreNB, 0, 256, pBin.valSeuil, &pBin.valSeuil, SeuilImageNB, (void*)&pBin);
    AjouteGlissiere("Seuil TYPE", nomFenetreNB, 0, 4, 0, &pBin.valTypeSeuil, SeuilImageNB, (void*)&pBin);
    namedWindow(nomFenetreNBAda);
    AjouteGlissiere("Methode", nomFenetreNBAda, ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C, pBin.valTypeSeuilAda, &pBin.valTypeSeuilAda, SeuilImageNBAdaptatif, (void*)&pBin);
    AjouteGlissiere("Taille des blocs", nomFenetreNBAda, 1, 1024, pBin.blockSize, &pBin.blockSize, SeuilImageNBAdaptatif, (void*)&pBin);
    imshow("original", pBin.mBGR);
    imshow("original NB", pBin.mNB);
    SeuilImageNB(pBin.valSeuil, (void*)&pBin);
    SeuilImageNBAdaptatif(pBin.blockSize, (void*)&pBin);
    char code=0;
    while (code != 27)
    {
        code = waitKey();
        switch (code) {
        case 'o':
            pBin.seuillageAdaptatif = false;
            pBin.valSeuil = saturate_cast<int>(threshold(pBin.mNB, dst, 0, 255, THRESH_BINARY | THRESH_OTSU));
            setTrackbarPos("Seuil", nomFenetreNB , pBin.valSeuil);
            SeuilImageNB(pBin.valSeuil, (void*)&pBin);
            break;
        case't':
            pBin.seuillageAdaptatif = false;
            pBin.valSeuil = saturate_cast<int>(threshold(pBin.mNB, dst, 0, 255, THRESH_BINARY | THRESH_TRIANGLE));
            setTrackbarPos("Seuil", nomFenetreNB, pBin.valSeuil);
            SeuilImageNB(pBin.valSeuil, (void*)&pBin);
            break;

        }

    }
}

void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut,int *valGlissiere, void(*f)(int, void *),void *r)
{
    createTrackbar(nomGlissiere, nomFenetre, valGlissiere, 1, f, r);
    setTrackbarMin(nomGlissiere, nomFenetre, minGlissiere);
    setTrackbarMax(nomGlissiere, nomFenetre, maxGlissiere);
    setTrackbarPos(nomGlissiere, nomFenetre, valeurDefaut);
}

void Seuillage(ParamBinarisation *pBin)
{
    if (pBin->lutRND.empty())
    {
        RNG r;
        pBin->lutRND = Mat::zeros(256, 1, CV_8UC3);
        r.fill(pBin->lutRND, RNG::UNIFORM, 0, 256);
    }
    Mat dst;
    String nomMethode;
	if (pBin->seuillageAdaptatif)
	{
		adaptiveThreshold(pBin->mNB, dst, 255, pBin->adaptiveMethod, pBin->valTypeSeuilAda, 2 * pBin->blockSize + 1, 0);
        nomMethode="adaptiveThreshold";
        imshow(nomFenetreNBAda, dst);
	}
	else
	{
		threshold(pBin->mNB, dst, static_cast<double>(pBin->valSeuil), 255, pBin->valTypeSeuil);
        nomMethode = "threshold";
		imshow(nomFenetreNB, dst);
	}
    Mat labels1, labels2, cc;

    double ccMin, ccMax;
    connectedComponents(dst, labels1, 8, CV_16U);
    minMaxIdx(labels1, &ccMin, &ccMax);
    bitwise_not(dst, dst);
    connectedComponents(dst, labels2, 8, CV_16U);
    Mat mask = labels1 == 0;
    add(labels2, ccMax, labels2, mask);
    labels1 = labels1 + labels2 - 1;
    minMaxIdx(labels1, &ccMin, &ccMax);
    labels1.convertTo(cc, CV_8UC3, 1, 0);
    vector<Mat> xx = { cc,cc,cc };
    merge(xx, cc);
    LUT(cc, pBin->lutRND, cc);
    //applyColorMap(cc, cc, *lutRND.get());
    putText(cc, nomMethode, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
    imshow("Labels", cc);

}

void SeuilImageNB(int x, void *r)
{
	ParamBinarisation *pBin = (ParamBinarisation*)r;
	pBin->seuillageAdaptatif = false;
	Seuillage(pBin);
}

void SeuilImageNBAdaptatif(int x, void *r)
{
    ParamBinarisation *pBin = (ParamBinarisation*)r;
	pBin->seuillageAdaptatif = true;
	Seuillage(pBin);
}
