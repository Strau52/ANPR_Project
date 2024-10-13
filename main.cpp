#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tesseract/baseapi.h>

#define _PRESENTATION
#define _DEBUG

using namespace cv;
using namespace std;

int main() {
    const string imagePath = "../../Plates/newPlateMegane.jpg";
    const auto image = imread(imagePath);

    if (image.empty()) {
        cerr << "Nem sikerült betölteni a képet: " << imagePath << endl;
        return -1;
    }

    Mat resizedImage;
    resize(image, resizedImage, Size(), 0.5, 0.5);

#ifdef _DEBUG
    imshow("Resized image", resizedImage);
    waitKey(0);
#endif

    // Szürkeárnyalatos kép
    Mat grayImage;
    cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);

#ifdef _DEBUG
    imshow("Gray image", grayImage);
    waitKey(0);
#endif

    // Gauss-szűrés
    Mat blurredImage;
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 0);

#ifdef _DEBUG
    imshow("Blurred image", blurredImage);
    waitKey(0);
#endif

    // Canny éldetektálás
    Mat edges;
    Canny(blurredImage, edges, 100, 200);

#ifdef _PRESENTATION
    imshow("Canny Edges", edges);
    waitKey(0);
#endif

    // Kontúrok keresése
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Rendszámhoz keresése
    Rect licensePlateRect;
    for (const auto &contour: contours) {
        Rect rect = boundingRect(contour);
        float aspectRatio;
        aspectRatio = static_cast<float>(rect.width) / rect.height;

        if (aspectRatio > 2.0 && aspectRatio < 5.0 && rect.width > 100 && rect.height > 25 && rect.height < 150) {
            licensePlateRect = rect;
            rectangle(resizedImage, licensePlateRect, Scalar(0, 255, 0), 2);

#ifdef _PRESENTATION
            imshow("Detected License Plate", resizedImage);
            waitKey(0);
#endif
            break;
        }
    }

    if (licensePlateRect.area() > 0) {
        Mat licensePlate = grayImage(licensePlateRect);

#ifdef _DEBUG
        imshow("License image", licensePlate);
        waitKey(0);
#endif

        // Rendszám elforgatása
        Point2f center(licensePlate.cols / 2.0, licensePlate.rows / 2.0);
        Mat rotationMatrix = getRotationMatrix2D(center, -15, 1.0); // Elforgatás 15 fokkal
        Mat rotatedLicensePlate;
        warpAffine(licensePlate, rotatedLicensePlate, rotationMatrix, licensePlate.size());

#ifdef _PRESENTATION
        imshow("Rotated License Plate", rotatedLicensePlate);
        waitKey(0);
#endif

        // Kép előkészítés
        Mat sharpenedImage;
        Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0); // Élesítő kernel
        filter2D(rotatedLicensePlate, sharpenedImage, -1, kernel);

        // Otsu thresholding
        Mat thresholdedImage;
        threshold(sharpenedImage, thresholdedImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // Karakterek kontúrok
        vector<vector<Point> > charContours;
        findContours(thresholdedImage, charContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        tesseract::TessBaseAPI ocr;
        if (ocr.Init("C:/Program Files/Tesseract-OCR/tessdata/", "eng")) {
            cerr << "Hiba az OCR inicializálásakor!" << endl;
            return -1;
        }
        ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-");


        string recognizedText;

        // Karakterek detektálása
        for (const auto &contour: charContours) {
            if (Rect charRect = boundingRect(contour); charRect.height > 25 && charRect.width > 10) {
                Mat charImage = thresholdedImage(charRect);

#ifdef _PRESENTATION
                imshow("Detected Character", charImage);
                waitKey(0);
#endif

                // OCR futtatás
                ocr.SetImage(charImage.data, charImage.cols, charImage.rows, 1, charImage.step);
                recognizedText += string(ocr.GetUTF8Text());
            }
        }

        // cout << to_string(recognizedText.size()) << endl;
        if (recognizedText.size() > 7) {
            recognizedText[2] = ' ';
        }

        cout << "Felismert rendszám: " << recognizedText << endl;

        ocr.End();
    } else {
        cerr << "Nem található rendszám!" << endl;
    }

    return 0;
}
