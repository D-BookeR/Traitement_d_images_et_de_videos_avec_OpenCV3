#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static void Glissiere(int x, void *r);

struct ImageHuile {
    String fenetreResultat = "Peinture Huile";
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
    ImageHuile p;
    VideoCapture v(0);
    v>> p.img;
    p.fenetreResultat="Original";
    namedWindow(p.fenetreResultat, WINDOW_KEEPRATIO);
    p.pas = 20;
    p.largeur = 3;
    createTrackbar("Pas", p.fenetreResultat, &p.pas, 1, Glissiere, &p);
    setTrackbarMin("Pas", p.fenetreResultat, 1);
    setTrackbarMax("Pas", p.fenetreResultat, 256);
    setTrackbarPos("Pas", p.fenetreResultat, p.pas);
    createTrackbar("Largeur", p.fenetreResultat, &p.largeur, 1, Glissiere, &p);
    setTrackbarMin("Largeur", p.fenetreResultat, 0);
    setTrackbarMax("Largeur", p.fenetreResultat, 100);
    setTrackbarPos("Largeur", p.fenetreResultat, p.largeur);
    while (waitKey(20) != 27)
    {
        v>>p.img;
        Glissiere(0,&p);

    }
    return 0;
}


class ParallelPeintureHuile : public ParallelLoopBody
{
private:
    Mat &imgSrc;
    Mat &dst;
    Mat &imgLuminance;
    int largeur;

public:
    ParallelPeintureHuile(Mat& img, Mat &d, Mat &iLuminance, int r) :
        imgSrc(img),
        dst(d),
        imgLuminance(iLuminance),
        largeur(r)
    {}
    virtual void operator()(const Range& range) const
    {
        vector<int> histogramme(256);
        vector<Vec3f> moyenneRGB(256);

        for (int y = range.start; y < range.end; y++)
        {
            Vec3b *vDst = (Vec3b *)dst.ptr(y);
            for (int x = 0; x < imgSrc.cols; x++, vDst++)
            {
                if (x == 0)
                {
                    histogramme.assign(256, 0);
                    moyenneRGB.assign(256, Vec3f(0, 0, 0));
                    for (int yy = -largeur; yy <= largeur; yy++)
                    {
                        if (y + yy >= 0 && y + yy < imgSrc.rows)
                        {
                            Vec3b *vPtr = (Vec3b *)imgSrc.ptr(y + yy) + x - 0;
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
                            Vec3b *vPtr = (Vec3b *)imgSrc.ptr(y + yy) + x - largeur - 1;
                            uchar *uc = imgLuminance.ptr(y + yy) + x - largeur - 1;
                            int xx = -largeur - 1;
                            if (x + xx >= 0 && x + xx < imgSrc.cols)
                            {
                                histogramme[*uc]--;
                                moyenneRGB[*uc] -= *vPtr;
                            }
                            vPtr = (Vec3b *)imgSrc.ptr(y + yy) + x + largeur;
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
    }
};

class DivisionEntiere {
    int q;
public:
    DivisionEntiere(int v) { q = v; };
    void operator ()(uchar &pixel, const int * position) const {
        pixel = pixel / q;
    }
};




void Glissiere(int x, void *r)
{
    ImageHuile *p = (ImageHuile *)r;
    TickMeter tm;
    tm.start();
    Mat dst(p->img.rows, p->img.cols, CV_8UC3);
    Mat imgLuminance;
    cvtColor(p->img, imgLuminance, COLOR_BGR2GRAY);
    if (!imgLuminance.isContinuous())
    {
        cout << "Parametre invalide\n";
        return;
    }
    imgLuminance.forEach<uchar>(DivisionEntiere(256 / p->pas));
    ParallelPeintureHuile algoPeinture(p->img, dst, imgLuminance, p->largeur);
    parallel_for_(Range(0, p->img.rows), algoPeinture, getNumThreads());
    tm.stop();
    if (dst.empty())
        cout << "Erreur dans la fonction PeintureHuile";
    else
    {
        cout << "Temps  = " << tm.getTimeSec() << "Pour un voisinage de taille " << 2 * p->largeur + 1 << " et un pas de " << p->pas;
        cout << " et une image de " << p->img.rows*p->img.cols << "\n";
        imshow(p->fenetreResultat, dst);
        imwrite(format("img_%d_%d.tiff", p->largeur, p->pas), dst);
    }
}
