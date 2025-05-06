#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

Vec3b bilinearInterpolate(const Mat& img, float x, float y) {
    int x1 = floor(x), y1 = floor(y);
    int x2 = min(x1 + 1, img.cols - 1);
    int y2 = min(y1 + 1, img.rows - 1);

    float dx = x - x1, dy = y - y1;

    Vec3b a = img.at<Vec3b>(y1, x1);
    Vec3b b = img.at<Vec3b>(y1, x2);
    Vec3b c = img.at<Vec3b>(y2, x1);
    Vec3b d = img.at<Vec3b>(y2, x2);

    Vec3b result;
    for (int i = 0; i < 3; ++i) {
        result[i] = saturate_cast<uchar>(
            (1 - dx) * (1 - dy) * a[i] +
            dx * (1 - dy) * b[i] +
            (1 - dx) * dy * c[i] +
            dx * dy * d[i]
        );
    }
    return result;
}

Mat escalar(const Mat& input, float factor, bool paralelo) {
    int newW = int(input.cols * factor);
    int newH = int(input.rows * factor);
    Mat output(newH, newW, input.type());

    #pragma omp parallel for if(paralelo)
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            float srcX = x / factor;
            float srcY = y / factor;
            output.at<Vec3b>(y, x) = bilinearInterpolate(input, srcX, srcY);
        }
    }

    return output;
}

Mat rotar(const Mat& input, float grados, bool paralelo) {
    float rad = grados * CV_PI / 180.0;
    float cosA = cos(rad), sinA = sin(rad);

    int w = input.cols, h = input.rows;

    // Nuevo tamaño tras rotación
    int newW = int(abs(w * cosA) + abs(h * sinA));
    int newH = int(abs(h * cosA) + abs(w * sinA));

    Mat output(newH, newW, input.type(), Scalar(0, 0, 0));

    float cxOld = w / 2.0f;
    float cyOld = h / 2.0f;
    float cxNew = newW / 2.0f;
    float cyNew = newH / 2.0f;

    #pragma omp parallel for if(paralelo)
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            // Coordenadas respecto al centro nuevo
            float dx = x - cxNew;
            float dy = y - cyNew;

            // Coordenadas originales aplicando rotación inversa
            float srcX =  dx * cosA + dy * sinA + cxOld;
            float srcY = -dx * sinA + dy * cosA + cyOld;

            if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h) {
                output.at<Vec3b>(y, x) = bilinearInterpolate(input, srcX, srcY);
            }
        }
    }

    return output;
}


int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "Uso: " << argv[0] << " <entrada> <salida> <escalar|rotar> <factor> <secuencial|paralelo>\n";
        return 1;
    }

    string entrada = argv[1];
    string salida = argv[2];
    string operacion = argv[3];
    float factor = stof(argv[4]);
    string modo = argv[5];

    bool paralelo = (modo == "paralelo");

    Mat imagen = imread(entrada);
    if (imagen.empty()) {
        cerr << "No se pudo cargar la imagen de entrada.\n";
        return 1;
    }

    Mat resultado;

    auto inicio = high_resolution_clock::now();

    if (operacion == "escalar") {
        resultado = escalar(imagen, factor, paralelo);
    } else if (operacion == "rotar") {
        resultado = rotar(imagen, factor, paralelo);
    } else {
        cerr << "Operación no válida. Usa 'escalar' o 'rotar'.\n";
        return 1;
    }

    auto fin = high_resolution_clock::now();
    auto duracion = duration_cast<milliseconds>(fin - inicio);

    imwrite(salida, resultado);
    cout << "Imagen procesada y guardada en: " << salida << "\n";
    cout << "Tiempo de procesamiento: " << duracion.count() << " ms\n";
    return 0;
}
