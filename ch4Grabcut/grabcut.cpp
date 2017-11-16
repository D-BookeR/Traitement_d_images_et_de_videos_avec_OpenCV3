#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

static void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r = NULL);
static void EffaceFenetre(String nomFenetre, Mat x);
static void DefRectangle(int event, int x, int y, int flags, void *userData);

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |     | chemin complet de l'image couleur (3 canaux)}";

struct ParamGrabCut {
    int rctEnCours;
    Rect r;
    Mat img;
    Mat result;
    int nbIteration;
};

String nomFenetre = "Image Originale";

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
    ParamGrabCut pgc;
    pgc.rctEnCours=0;
    pgc.nbIteration=1;
    pgc.img = imread(nomFic, IMREAD_COLOR);
    if (pgc.img.empty())
    {
        cout << "Fichier vide ou inexistant. Vérifiez le nom du fichier.\n";
        return 0;
    }
    imshow(nomFenetre,pgc.img);
    AjouteGlissiere("iter", nomFenetre, 1, 10, pgc.nbIteration, &pgc.nbIteration, NULL, NULL);

    setMouseCallback(nomFenetre,DefRectangle, &pgc);
    Mat maskTravail,maskOriginal;
    Mat bk,fk;
	bool initGrabCut = false;
    maskOriginal =Mat(pgc.img.size(),CV_8UC1,Scalar::all(cv::GC_BGD));
    int code=0;
    do
    {
        code =  waitKey(30)&0xFF;
        switch (code) {
        case 'r':
            pgc.r=Rect(0,0,0,0);
            maskOriginal = Mat(pgc.img.size(), CV_8UC1, Scalar::all(cv::GC_BGD));
            pgc.result = Mat::zeros(pgc.img.size(), pgc.img.type());
            maskTravail=Mat();
			initGrabCut = false;
            imshow("grabcut", pgc.result);
            imshow(nomFenetre, pgc.img - 0.5* pgc.result);
            break;
		case '+':
            if (maskTravail.empty())
            {
                maskOriginal(pgc.r) = cv::GC_PR_FGD;
                if (pgc.r.area() != 0)
                    initGrabCut = true;
            }
            else
            {
                maskOriginal(pgc.r) = cv::GC_FGD;
                maskTravail(pgc.r) = cv::GC_FGD;
            }
			break;
		case '-':
            maskOriginal(pgc.r) = cv::GC_BGD;
            if (!maskTravail.empty())
                maskTravail(pgc.r) = cv::GC_BGD;
            break;
        case 'g':
            if (initGrabCut)
            {
                EffaceFenetre("grabcut", pgc.img);
                maskOriginal.copyTo(maskTravail);
                grabCut(pgc.img, maskTravail, Rect(), bk, fk, pgc.nbIteration, GC_INIT_WITH_MASK);
            }
            break;
        case 'c':
            if (initGrabCut)
            {
                EffaceFenetre("grabcut", pgc.img);
                if (maskTravail.empty())
                    maskOriginal.copyTo(maskTravail);
                else
                {
                    pgc.nbIteration++;
                    setTrackbarPos("iter", nomFenetre, pgc.nbIteration);
                }
                grabCut(pgc.img, maskTravail, pgc.r, bk, fk, 1, GC_EVAL);
            }
            break;
        }
        if (initGrabCut && (code == 'c' || code == 'g'))
        {
            cv::Mat mask = (maskTravail == cv::GC_FGD) | (maskTravail == cv::GC_PR_FGD);
            pgc.result = Mat::zeros(pgc.img.size(), pgc.img.type());
            pgc.img.copyTo(pgc.result, mask);
            imshow("grabcut", pgc.result);
            imshow(nomFenetre, pgc.img - 0.5* pgc.result);
        }
    }
    while (code!=27);
}

void DefRectangle(int event, int x, int y, int flags, void *userData)
{
    ParamGrabCut *pgc = (ParamGrabCut*)userData;
    if (flags == EVENT_FLAG_LBUTTON)
    {
        if (pgc->rctEnCours == 0)
        {
            pgc->r.x = x;
            pgc->r.y = y;
            pgc->r.width = 0;
            pgc->r.height = 0;
            pgc->rctEnCours = 1;
        }
        else if (pgc->rctEnCours == 1)
        {
            Point tl = pgc->r.tl(), br = pgc->r.br();
            if (x != pgc->r.x)
            {
                if (x < pgc->r.x)
                {
                    pgc->r.x = x;
                    pgc->r.width = br.x - x - 1;
                }
                else
                    pgc->r.width = x - tl.x - 1;

            }
            if (y != pgc->r.y)
            {
                if (y < pgc->r.y)
                {
                    pgc->r.y = y;
                    pgc->r.height = br.y - y - 1;
                }
                else
                    pgc->r.height = y - tl.y - 1;

            }
            if (pgc->r.br().x > pgc->img.size().width)
            {
                pgc->r.width = pgc->img.size().width - pgc->r.x;
            }
            if (pgc->r.br().y > pgc->img.size().height)
            {
                pgc->r.height = pgc->img.size().height - pgc->r.y;
            }
        }
    }
    else if (event == EVENT_LBUTTONUP && pgc->rctEnCours == 1)
    {
        pgc->rctEnCours = 0;
    }
    Mat img = pgc->img.clone();
    rectangle(img, pgc->r, Scalar(0, 255, 0), 2);
    imshow(nomFenetre, img - 0.5* pgc->result);
    waitKey(10);
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
    Mat y = Mat::zeros(x.size(), CV_8UC1);
    putText(y, "Grabcut en cours!", Point(10, x.rows / 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255));
    imshow(nomFenetre, y);
    waitKey(1);
}
