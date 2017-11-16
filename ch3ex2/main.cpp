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
    String nomFic = parser.get<String>(0);
    if (nomFic.length() == 0)
    {
        parser.printMessage();
        return 0;
    }
    ImageHuile p;
    p.img = imread(nomFic, IMREAD_COLOR);
    if (p.img.empty())
    {
        cout << "L'image est vide ;  vérifier le chemin des données.\n";
        parser.printMessage();
        return 0;
    }
    if (parser.has("z") && (p.img.rows>256 || p.img.cols>256))
        resize(p.img, p.img, Size(), min(256. / p.img.rows, 256. / p.img.cols), min(256. / p.img.rows, 256. / p.img.cols));
    namedWindow(p.fenetreResultat, WINDOW_KEEPRATIO);
    setWindowProperty("Original", WND_PROP_ASPECT_RATIO, WINDOW_KEEPRATIO);
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

    imshow("Original", p.img);
    waitKey(0);
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
    cvtColor(p->img, imgLuminance, CV_BGR2GRAY);
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
















// Parallel execution with function object.
typedef Vec3b Pixelp;

int v = 32;
int ff(uchar x)
{
    return (x / v)*v;
}

void ffini(Pixelp &p, const int * position) 
{
    p[0] = ff(p[0]);
    p[1] = ff(p[1]);
    p[2] = ff(p[2]);
}

typedef cv::Point3_<uint8_t> Pixel;
// Parallel execution with function object.
class Operator {
    int pas;
    
public :
    Operator(int v){pas=v;};
    void operator ()(Pixel &pixel, const int * position) const {
        pixel.x = 255/pas;
    }
};

    void MyFunction(Pixel &pixel, const int * position) 
    {
        pixel.x = 255;
    }




