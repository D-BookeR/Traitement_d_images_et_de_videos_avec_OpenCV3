#include <opencv2/opencv.hpp> 
#include <opencv2/text.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace text;
using namespace std;

void AjouteGlissiere(String nomGlissiere, String nomFenetre, int minGlissiere, int maxGlissiere, int valeurDefaut, int *valGlissiere, void(*f)(int, void *), void *r);
void MAJImage(int, void *r);

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{@arg1                |texte1.png     | nom de l'image }";

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String nomFichierImage = parser.get<String>(0);

    Mat mOriginal=imread(nomFichierImage,IMREAD_COLOR);
    if (mOriginal.empty())
    {
        cout<< "Image vide!\n";
        return -1;
    }
    String nomFenetre="Texte dans image";
    namedWindow(nomFenetre);
    int nivBruitMax = 100;
    int majImage = 1;
    int majOCR = 1;
    int oem=3;
    int psm=3;
    int indLangue=1;
    int niveau=1;
    vector<string> langue = { "eng","fra" };
    vector<string> niveauAnalyse = { "mot","ligne" };
    vector<string> formePage={
        "Orientation and script detection(OSD) only.",
        "Automatic page segmentation with OSD.",
        "Automatic page segmentation, but no OSD, or OCR.",
        "Fully automatic page segmentation, but no OSD. (Default)",
        "Assume a single column of text of variable sizes.",
        "Assume a single uniform block of vertically aligned text.",
        "Assume a single uniform block of text.",
        "Treat the image as a single text line.",
        "Treat the image as a single word.",
        "Treat the image as a single word in a circle.",
        "Treat the image as a single character." };
    vector<int> niveauComposant ={text::OCR_LEVEL_WORD ,text::OCR_LEVEL_TEXTLINE };
    vector<string> moteurOCR={"Original Tesseract only.","Neural nets LSTM only.","Tesseract + LSTM.","Default, based on what is available."};
    AjouteGlissiere("maxBruit", nomFenetre, 0, 200, nivBruitMax, &nivBruitMax, MAJImage, &majImage);
    AjouteGlissiere("oem", nomFenetre, 0, 4, oem, &oem, MAJImage, &majOCR);
    AjouteGlissiere("psm", nomFenetre, 0, 10, psm, &psm, MAJImage, &majOCR);
    AjouteGlissiere("lang", nomFenetre, 0, 1, indLangue, &indLangue, MAJImage, &majOCR);
    AjouteGlissiere("Niveau", nomFenetre, 0, 1, niveau, &niveau, NULL, NULL);
    int code=0;
    Ptr<text::BaseOCR> ocr;
    Mat m;
    fstream fEfface;
    fEfface.open("ocrResultats.txt", std::fstream::out);
    fEfface.close();
    do
    {
        if (majImage)
        {
            Mat m2(mOriginal.rows, mOriginal.cols, CV_8UC3, Scalar::all(0));
            RNG r;
            if (nivBruitMax>0)
            {
                r.fill(m2, RNG::UNIFORM, 0, nivBruitMax);
                m = mOriginal + m2;
            }
            else
                m = mOriginal.clone();
            imshow(nomFenetre, m);
            waitKey(10);
            majImage =0;
        }
        if (code == 'r')
        {
            if (majOCR)
            {
                ocr = text::OCRTesseract::create(NULL, langue[indLangue].c_str(), NULL, oem, psm);
                majOCR=0;
            }
            vector<Rect> rectMot;
            string textEntier;
            vector<string> text;
            vector<float> proba;

            fstream fs;
            fs.open("ocrResultats.txt", std::fstream::app);
            cout << "****************************************\n";
            cout << "EXTRACTION TEXTE\n";
            cout << "langue :" << langue[indLangue] << "\n";
            cout << "moteur : " << moteurOCR[oem] << "\n";
            cout << "analyse : " << formePage[psm] << "\n";
            cout << "niveau : " << niveauAnalyse[niveau] << "\n";
            cout << "image : " << nomFichierImage << "\n";
            fs << "****************************************\n";
            fs << "EXTRACTION TEXTE\n";
            fs << "langue :" << langue[indLangue] << "\n";
            fs << "moteur : " << moteurOCR[oem] << "\n";
            fs << "analyse : " << formePage[psm] << "\n";
            fs << "niveau : " << niveauAnalyse[niveau] << "\n";
            fs << "image : " << nomFichierImage << "\n";
            ocr->run(m, textEntier, &rectMot, &text, &proba, niveauComposant[niveau]);
            cout << "Texte \n" << textEntier << "\n";
            fs << "Texte \n" << textEntier << "\n";
            for (int i = 0; i < rectMot.size(); i++)
                cout << "proba = " << proba[i] << "->" << rectMot[i] << " : " << text[i] << "\n";
        }
        code=waitKey(20);

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

void MAJImage(int, void *r)
{
    *((int *)r) = 1;
}

