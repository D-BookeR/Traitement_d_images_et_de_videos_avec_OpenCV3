#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

static int DivisionEntiere(Mat x, int q);
static Mat PeintureHuile(Mat imgSrc, int largeur, int pasLuminance);
static void Glissiere(int x, void *r);

struct ImageHuile {
    String fenetreResultat="Peinture Huile";
    int largeur;
    int pas;
    Mat img;
};

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{z                    |     | réduire les images }"
"{@arg1                |     | chemin complet de l'image couleur (3 canaux)}"
;


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
    ImageHuile p;
    p.img=imread(nomFic, IMREAD_COLOR);
    if (p.img.empty())
    {
        cout <<"L'image est vide ;  vérifier le chemin des données.\n";
        parser.printMessage();
        return 0;
    }
    if (parser.has("z") &&(p.img.rows>256 || p.img.cols>256))
        resize(p.img,p.img,Size(),min(256./p.img.rows, 256. / p.img.cols),  min(256. / p.img.rows, 256. / p.img.cols));
    namedWindow(p.fenetreResultat, WINDOW_KEEPRATIO );
    setWindowProperty("Original", WND_PROP_ASPECT_RATIO, WINDOW_KEEPRATIO);
    p.pas = 20;
    p.largeur = 3;
    createTrackbar("Pas", p.fenetreResultat,&p.pas,1,Glissiere,&p);
    setTrackbarMin("Pas", p.fenetreResultat, 1);
    setTrackbarMax("Pas", p.fenetreResultat, 256);
    setTrackbarPos("Pas", p.fenetreResultat, p.pas);
    createTrackbar("Largeur", p.fenetreResultat, &p.largeur, 1,Glissiere, &p);
    setTrackbarMin("Largeur", p.fenetreResultat, 0);
    setTrackbarMax("Largeur", p.fenetreResultat, 100);
    setTrackbarPos("Largeur", p.fenetreResultat, p.largeur);

    imshow("Original", p.img);
    waitKey(0);
    return 0;
}

Mat PeintureHuile(Mat imgSrc, int largeur, int pasLuminance)
{
    vector<int> histogramme;
    vector<Vec3f> moyenneRGB;
    Mat dst(imgSrc.rows, imgSrc.cols, CV_8UC3);
    Mat imgLuminance;
    cvtColor(imgSrc, imgLuminance, CV_BGR2GRAY);
    if (DivisionEntiere(imgLuminance, 256/pasLuminance))
        return Mat();
    for (int y = 0; y < imgSrc.rows; y++)
    {
        Vec3b *vDst = dst.ptr<Vec3b>(y);
        for (int x = 0; x < imgSrc.cols; x++, vDst++) //for each pixel
        {
            if (x == 0)
            {
                histogramme.assign(256, 0);
                moyenneRGB.assign(256, Vec3f(0, 0, 0));
                for (int yy = -largeur; yy <= largeur; yy++)
                {
                    if (y + yy >= 0 && y + yy < imgSrc.rows)
                    {
                        Vec3b *vPtr = imgSrc.ptr<Vec3b>(y + yy) + x - 0;
                        uchar *uc = imgLuminance.ptr(y + yy) + x - 0;
                        for (int xx = 0; xx <= largeur; xx++, vPtr++, uc++)
                        {
                            if (x + xx >= 0 && x + xx < imgSrc.cols)
                            {
                                histogramme[*uc]++;
                                moyenneRGB[*uc] += *vPtr;
                            }
                        }
                    }
                }

            }
            else
            {
                for (int yy = -largeur; yy <= largeur; yy++)
                {
                    if (y + yy >= 0 && y + yy < imgSrc.rows)
                    {
                        Vec3b *vPtr = imgSrc.ptr<Vec3b>(y + yy) + x - largeur - 1;
                        uchar *uc = imgLuminance.ptr(y + yy) + x - largeur - 1;
                        int xx = -largeur - 1;
                        if (x + xx >= 0 && x + xx < imgSrc.cols)
                        {
                            histogramme[*uc]--;
                            moyenneRGB[*uc] -= *vPtr;
                        }
                        vPtr = imgSrc.ptr<Vec3b>(y + yy) + x + largeur;
                        uc = imgLuminance.ptr(y + yy) + x + largeur;
                        xx = largeur;
                        if (x + xx >= 0 && x + xx < imgSrc.cols)
                        {
                            histogramme[*uc]++;
                            moyenneRGB[*uc] += *vPtr;
                        }
                    }
                }
            }
            int64 pos = distance(histogramme.begin(), max_element(histogramme.begin(), histogramme.end()));
            *vDst = moyenneRGB[pos] / histogramme[pos];
        }
    }
    return dst;
}

int DivisionEntiere(Mat x, int q)
{
    if (q == 0)
        return 1;
    if (x.type() != CV_8UC1)
        return 2;
    if (!x.isContinuous())
        return 3;
    int nbPixels= x.rows*x.cols;
    uchar *ptr = x.ptr(0);
    for (int i = 0; i < nbPixels; i++,ptr++)
    {
        *ptr = *ptr / q;
    }
    return 0;
}

static void Glissiere(int x, void *r)
{
    ImageHuile *p = (ImageHuile *)r;
    int64 tIni=getTickCount();
    Mat dst = PeintureHuile(p->img, p->largeur, p->pas);
    int64 tFin = getTickCount();
    if (dst.empty())
        cout << "Erreur dans la fonction PeintureHuile";
    else
    {
        cout<<"Temps  = "<<double(tFin-tIni)/getTickFrequency()<<"Pour un voisinage de taille "<< 2*p->largeur+1<< " et un pas de "<<p->pas;
        cout<<" et une image de "<<p->img.rows*p->img.cols<<"\n";
        imshow(p->fenetreResultat, dst);
        imwrite(format("img_%d_%d.tiff", p->largeur, p->pas), dst);
    }
}
