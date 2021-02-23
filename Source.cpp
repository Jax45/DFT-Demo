#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
//Jackson Hoenig implementing https://github.com/Saleh-I/Filtering-in-frequency-domain all rights for code used go to this github.
using namespace cv;

void fftshift(const Mat& input_img, Mat& output_img)
{
	output_img = input_img.clone();
	int cx = output_img.cols / 2;
	int cy = output_img.rows / 2;
	Mat q1(output_img, Rect(0, 0, cx, cy));
	Mat q2(output_img, Rect(cx, 0, cx, cy));
	Mat q3(output_img, Rect(0, cy, cx, cy));
	Mat q4(output_img, Rect(cx, cy, cx, cy));

	Mat temp;
	q1.copyTo(temp);
	q4.copyTo(q1);
	temp.copyTo(q4);
	q2.copyTo(temp);
	q3.copyTo(q2);
	temp.copyTo(q3);
}

void calculateDFT2(Mat& scr, Mat& dst) {
	cv::Mat fft;
	dft(scr, fft, DFT_COMPLEX_OUTPUT);

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(scr.rows);
	int n = getOptimalDFTSize(scr.cols); // on the border add zero values
	copyMakeBorder(scr, padded, 0, m - scr.rows, 0, n - scr.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	magI.copyTo(dst);
}

void calculateDFT(Mat& scr, Mat& dst)
{
	// define mat consists of two mat, one for real values and the other for complex values
	Mat planes[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);
	dst = complexImg;
}

Mat construct_H(Mat& scr, String type, float D0)
{
	Mat H(scr.size(), CV_32F, Scalar(1));
	float D = 0;
	if (type == "Ideal")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
				if (D > D0)
				{
					H.at<float>(u, v) = 0;
				}
			}
		}
		return H;
	}
	else if (type == "Gaussian")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
				H.at<float>(u, v) = exp(-D * D / (2 * D0 * D0));
			}
		}
		return H;
	}
}


void filtering(Mat& scr, Mat& dst, Mat& H)
{
	fftshift(H, H);
	Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

	Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	split(scr, planes_dft);

	Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
	planes_out[0] = planesH[0].mul(planes_dft[0]);
	planes_out[1] = planesH[1].mul(planes_dft[1]);

	merge(planes_out, 2, dst);

}


int main()
{
	Mat img = imread("./lena_gaussian.png", IMREAD_GRAYSCALE);
	Mat imgIn = imread("./lena_gaussian.png", 0);
	imgIn.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

	//displaydft
	Mat DFT_image2;
	calculateDFT2(imgIn, DFT_image2);

	/*imshow("DFT of image", DFT_image);
	waitKey();*/
	// construct H
	Mat H;
	H = construct_H(imgIn, "Gaussian", 25);
	Mat filter;
	H.copyTo(filter);

	// filtering
	Mat complexIH;
	filtering(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	//dft(DFT_image, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);

	imshow("original image", img);
	//normalize(DFT_image, DFT_image, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	imshow("DFT of image", DFT_image2);
	imshow("filter H", filter);
	//imshow("complexIH for filtering", complexIH);
	imshow("img out", imgOut);
	waitKey(0);
	//cv::imwrite("input_gray.jpg", img);
	//cv::imwrite("DFT.jpg", DFT_image2);
	cv::imwrite("d25Filter.jpg", filter);
	cv::imwrite("filteredImage.25.jpg", imgOut);

	return 0;
}



//using namespace cv;
//using namespace std;
//static void help(char** argv)
//{
//    cout << endl
//        << "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
//        << "The dft of an image is taken and it's power spectrum is displayed." << endl << endl
//        << "Usage:" << endl
//        << argv[0] << " [image_name -- default lena.jpg]" << endl << endl;
//}
//int main(int argc, char** argv)
//{
//    help(argv);
//    const char* filename = argc >= 2 ? argv[1] : "./Square.jpg";
//    Mat image = imread("./Square.jpg", IMREAD_GRAYSCALE);
//    if (image.empty()) {
//        cout << "Error opening image" << endl;
//        return EXIT_FAILURE;
//    }
    //cv::Mat fft;
    //dft(image, fft, DFT_COMPLEX_OUTPUT);

    //Mat padded;                            //expand input image to optimal size
    //int m = getOptimalDFTSize(I.rows);
    //int n = getOptimalDFTSize(I.cols); // on the border add zero values
    //copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    //Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    //Mat complexI;
    //merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    //dft(complexI, complexI);            // this way the result may fit in the source matrix
    //// compute the magnitude and switch to logarithmic scale
    //// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    //split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    //magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    //Mat magI = planes[0];
    //magI += Scalar::all(1);                    // switch to logarithmic scale
    //log(magI, magI);
    //// crop the spectrum, if it has an odd number of rows or columns
    //magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    //// rearrange the quadrants of Fourier image  so that the origin is at the image center
    //int cx = magI.cols / 2;
    //int cy = magI.rows / 2;
    //Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    //Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    //Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    //Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    //Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    //q0.copyTo(tmp);
    //q3.copyTo(q0);
    //tmp.copyTo(q3);
    //q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    //q2.copyTo(q1);
    //tmp.copyTo(q2);
    //normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
    //                                        // viewable image form (float between values 0 and 1).

   // 
   // imshow("Input Image", I);    // Show the result
   // imshow("spectrum magnitude", magI);
   // waitKey();

   // //call idft
   // Mat inverseImage;
   //// Mat padded;                            //expand input image to optimal size
   // m = getOptimalDFTSize(magI.rows);
   // n = getOptimalDFTSize(magI.cols); // on the border add zero values
   // copyMakeBorder(magI, padded, 0, m - magI.rows, 0, n - magI.cols, BORDER_CONSTANT, Scalar::all(0));
   // Mat planes2[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
   // //Mat complexI;
   // merge(planes2, 2, complexI);         // Add to the expanded another plane with zeros
   // idft(complexI, complexI);            // this way the result may fit in the source matrix
   // // compute the magnitude and switch to logarithmic scale
   // // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
   // split(complexI, planes2);                   // planes2[0] = Re(DFT(I), planes2[1] = Im(DFT(I))
   // magnitude(planes2[0], planes2[1], planes2[0]);// planes2[0] = magnitude
   // Mat magI2 = planes2[0];
   // magI2 += Scalar::all(1);                    // switch to logarithmic scale
   // log(magI2, magI2);
   // // crop the spectrum, if it has an odd number of rows or columns
   // magI2 = magI2(Rect(0, 0, magI2.cols & -2, magI2.rows & -2));
   // // rearrange the quadrants of Fourier image  so that the origin is at the image center
   // cx = magI2.cols / 2;
   // cy = magI2.rows / 2;
   // Mat q02(magI2, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
   // Mat q12(magI2, Rect(cx, 0, cx, cy));  // Top-Right
   // Mat q22(magI2, Rect(0, cy, cx, cy));  // Bottom-Left
   // Mat q32(magI2, Rect(cx, cy, cx, cy)); // Bottom-Right
   // Mat tmp2;                           // swap quadrants (Top-Left with Bottom-Right)
   // q02.copyTo(tmp2);
   // q32.copyTo(q02);
   // tmp2.copyTo(q32);
   // q12.copyTo(tmp2);                    // swap quadrant (Top-Right with Bottom-Left)
   // q22.copyTo(q12);
   // tmp2.copyTo(q22);
   // normalize(magI2, magI2, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
   //                                         // viewable image form (float between values 0 and 1).
   // imshow("inversed dft image", magI2);
   // waitKey();
//    return EXIT_SUCCESS;
//}