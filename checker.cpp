
#include <fstream>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <opencv2/opencv.hpp>


////////////////////////////////////////////////////////////
/// \brief calRangeCalib
/// \param _coorShift //actual_coor = _GPScoor - _coorShift  should be 3x1 cv::Mat
/// \param _Lidarcoor //should be 3Xn cv::Mat
/// \param _GPScoor   //should be 3X1 cv::Mat
/// \param _rangeErr  //the range measure error
///
void calRangeCalib( cv::Mat& _coorShift,
                    cv::Mat& _Lidarcoor,
                    cv::Mat& _GPScoor,
                    double& _rangeErr){
    cv::Mat actual_coor = _GPScoor - _coorShift;
    cv::Mat actual_coor_pow;
    cv::pow(actual_coor, 2, actual_coor_pow);
    double d_con = std::sqrt(cv::sum(actual_coor_pow)[0]);

    cv::Mat Lidar_coor_pow;
    cv::pow(_Lidarcoor, 2, Lidar_coor_pow);
    double d_lidar = std::sqrt(cv::sum(Lidar_coor_pow.col(0))[0] +
                     cv::sum(Lidar_coor_pow.col(1))[0] +
                     cv::sum(Lidar_coor_pow.col(2))[0])/_Lidarcoor.rows;

    _rangeErr = std::fabs(d_lidar - d_con);
}

///////////////////////////////////////////////////////////////////
/// \brief calBeamDirectionCalib
/// \param _coorShift    //gps_coor = _GPScoor * _transMatrix + _coorShift
/// \param _transMatrix  //3X3 cv::Mat
/// \param _Lidarcoor    //3xn cv::Mat
/// \param _GPScoor      //3X1 cv::Mat
/// \param _beamErr      //error value
///
void calBeamDirectionCalib( cv::Mat& _coorShift,
                            cv::Mat& _transMatrix,
                            cv::Mat& _Lidarcoor,
                            cv::Mat& _GPScoor,
                            double& _beamErr){
    cv::Mat gps_coor = _GPScoor * _transMatrix + _coorShift;
    cv::Mat lidar_coor = cv::Mat::zeros(3, 1, CV_64F);
    for(size_t i = 0; i < 3; i++)
        lidar_coor.at<double>(i) = cv::mean(_Lidarcoor.col(i))[0];

    double theta_lidar = std::acos(lidar_coor.at<double>(0)/
                                   std::sqrt(std::pow(lidar_coor.at<double>(0), 2)+
                                   std::pow(lidar_coor.at<double>(1), 2)));
    double v_lidar = std::asin(lidar_coor.at<double>(1)/
                               std::sqrt(std::pow(lidar_coor.at<double>(0), 2)+
                               std::pow(lidar_coor.at<double>(1), 2)));

    double theta_gps = std::acos(gps_coor.at<double>(0)/
                                   std::sqrt(std::pow(gps_coor.at<double>(0), 2)+
                                   std::pow(gps_coor.at<double>(1), 2)));
    double v_gps = std::asin(gps_coor.at<double>(1)/
                               std::sqrt(std::pow(gps_coor.at<double>(0), 2)+
                               std::pow(gps_coor.at<double>(1), 2)));

    _beamErr = std::sqrt(std::pow(theta_gps - theta_lidar, 2) + std::pow(v_lidar - v_gps, 2));
}


/////////////////////////////////////////////////
/// \brief calEquOriginCalib
/// \param _coorShift
/// \param _GPSshift
/// \param _GPScoor
/// \param _devValue
///
void calEquOriginCalib(cv::Mat& _coorShift,
                  cv::Mat& _GPSshift,
                  cv::Mat& _GPScoor,
                  cv::Mat& _devValue){
    cv::Mat gps_coor = _GPScoor + _GPSshift;
    _devValue = gps_coor - _coorShift;
}

////////////////////////////////////////////////////
/// \brief calReferAxisCalib
/// \param _eupOrigincoor
/// \param _Lidarcoor
/// \param _GPScoor
/// \param _devAxisValue
///
void calReferAxisCalib(cv::Mat& _eupOrigincoor,
                  cv::Mat& _Lidarcoor,
                  cv::Mat& _GPScoor,
                  cv::Mat& _devAxisValue){
    double lidar_normal = std::sqrt(std::pow(_Lidarcoor.at<double>(0) - _eupOrigincoor.at<double>(0), 2) +
                                    std::pow(_Lidarcoor.at<double>(1) - _eupOrigincoor.at<double>(1), 2));
    double lidar_theta_x = std::acos((_Lidarcoor.at<double>(0) - _eupOrigincoor.at<double>(0))/lidar_normal);
    double lidar_theta_y = std::acos((_Lidarcoor.at<double>(1) - _eupOrigincoor.at<double>(1))/lidar_normal);
    double lidar_theta_z = std::acos((_Lidarcoor.at<double>(2) - _eupOrigincoor.at<double>(2))/lidar_normal);

    double gps_normal = std::sqrt(std::pow(_GPScoor.at<double>(0) - _eupOrigincoor.at<double>(0), 2) +
                                    std::pow(_GPScoor.at<double>(1) - _eupOrigincoor.at<double>(1), 2));
    double gps_theta_x = std::acos((_GPScoor.at<double>(0) - _eupOrigincoor.at<double>(0))/lidar_normal);
    double gps_theta_y = std::acos((_GPScoor.at<double>(1) - _eupOrigincoor.at<double>(1))/lidar_normal);
    double gps_theta_z = std::acos((_GPScoor.at<double>(2) - _eupOrigincoor.at<double>(2))/lidar_normal);

    double delta_theta_x = lidar_theta_x - gps_theta_x;
    double delta_theta_y = lidar_theta_y - gps_theta_y;
    double delta_theta_z = lidar_theta_z - gps_theta_z;

    _devAxisValue =  cv::Mat::zeros(3, 1, CV_64F);
    _devAxisValue.at<double>(0) = delta_theta_x;
    _devAxisValue.at<double>(1) = delta_theta_y;
    _devAxisValue.at<double>(2) = delta_theta_z;
}




///////////////////////////////////////
/// \brief calEquCoor_GPSCoor
/// \param _equcoor
/// \param _GPScoor
/// \param _coorShift
/// \param _transMatri
/// \param residualErr
///
void calEquCoor_GPSCoor(cv::Mat& _equcoor,
                        cv::Mat& _GPScoor,
                        cv::Mat& _coorShift,
                        cv::Mat& _transMatri,
                        cv::Mat& _residualErr)
{
    assert(_equcoor.rows == _GPScoor.rows);

    cv::Mat X, Y, Z, X2, Y2, Z2, R;
    X = _GPScoor.col(0).clone();
    Y = _GPScoor.col(1).clone();
    Z = _GPScoor.col(2).clone();
    X2 = _equcoor.col(0).clone();
    Y2 = _equcoor.col(1).clone();
    Z2 = _equcoor.col(2).clone();

    cv::Mat X2_pow, Y2_pow, Z2_pow;
    cv::pow(X2, 2, X2_pow);
    cv::pow(Y2, 2, Y2_pow);
    cv::pow(Z2, 2, Z2_pow);

    R = X2_pow + Y2_pow + Z2_pow;
    cv::sqrt(R, R);


    cv::Mat X_start = X.clone(), Y_start = Y.clone(), Z_start = Z.clone();
    cv::Mat X_obj = X2.clone(), Y_obj = Y2.clone(), Z_obj = Z2.clone();

    double X_delta = 0, Y_delta = 0, Z_delta = 0;
    double a = 0, b = 0, c = 0;


    //inital
    cv::Mat B_eof = cv::Mat::zeros(_equcoor.rows * 3, 6, CV_64F);
    cv::Mat L_norm = cv::Mat::zeros(_equcoor.rows * 3, 1, CV_64F);
    cv::Mat AA = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat BB = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat CC = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat X_val = cv::Mat::ones(6, 1, CV_64F) * 10;


    //loop
    unsigned int loop_counter = 0;
    while(cv::norm(X_val, 2) > 1e-7){

        double R_norm = 1 + a*a + b*b + c*c;
        cv::Mat R_array = cv::Mat(3, 3, CV_64F);
        R_array.at<double>(0, 0) = 1 + a*a - b*b - c*c; R_array.at<double>(0, 1) = -2*c -2*a*b;         R_array.at<double>(0, 2) = -2*b+2*a*c;
        R_array.at<double>(0, 1) = 2*c - 2*b*a;         R_array.at<double>(1, 1) = 1 - a*a + b*b - c*c; R_array.at<double>(1, 2) = -2*a-2*b*c;
        R_array.at<double>(0, 2) = 2*b + 2*a*c;         R_array.at<double>(2, 1) = 2*a - 2*b*c;         R_array.at<double>(2, 2) = 1-a*a-b*b+c*c;

        R_array = R_array / R_norm;


        for(int i = 0; i < _equcoor.rows; i++){
            AA.at<double>(0) = 2*a/R_norm*(X_delta-X_obj.at<double>(i))+2/R_norm*(a*X_start.at<double>(i)-b*Y_start.at<double>(i)+c*Z_start.at<double>(i));
            AA.at<double>(1) = 2*a/R_norm*(Y_delta-Y_obj.at<double>(i))+2/R_norm*(-b*X_start.at<double>(i)-a*Y_start.at<double>(i)-1*Z_start.at<double>(i));
            AA.at<double>(2) = 2*a/R_norm*(Z_delta-Z_obj.at<double>(i))+2/R_norm*(c*X_start.at<double>(i)+1*Y_start.at<double>(i)-a*Z_start.at<double>(i));
            BB.at<double>(0) = 2*b/R_norm*(X_delta-X_obj.at<double>(i))+2/R_norm*(-b*X_start.at<double>(i)-a*Y_start.at<double>(i)-1*Z_start.at<double>(i));
            BB.at<double>(1) = 2*b/R_norm*(Y_delta-Y_obj.at<double>(i))+2/R_norm*(-a*X_start.at<double>(i)+b*Y_start.at<double>(i)-c*Z_start.at<double>(i));
            BB.at<double>(2) = 2*b/R_norm*(Z_delta-Z_obj.at<double>(i))+2/R_norm*(1*X_start.at<double>(i)-c*Y_start.at<double>(i)-b*Z_start.at<double>(i));
            CC.at<double>(0) = 2*c/R_norm*(X_delta-X_obj.at<double>(i))+2/R_norm*(-c*X_start.at<double>(i)-1*Y_start.at<double>(i)+a*Z_start.at<double>(i));
            CC.at<double>(1) = 2*c/R_norm*(Y_delta-Y_obj.at<double>(i))+2/R_norm*(1*X_start.at<double>(i)-c*Y_start.at<double>(i)+b*Z_start.at<double>(i));
            CC.at<double>(2) = 2*c/R_norm*(Z_delta-Z_obj.at<double>(i))+2/R_norm*(a*X_start.at<double>(i)-b*Y_start.at<double>(i)+c*Z_start.at<double>(i));
            B_eof.at<double>((i)*3+0,0) = 1; B_eof.at<double>((i)*3+0,3) = AA.at<double>(0); B_eof.at<double>((i)*3+0,4) =  BB.at<double>(0); B_eof.at<double>((i)*3+0,5) = CC.at<double>(0);
            B_eof.at<double>((i)*3+1,1) = 1; B_eof.at<double>((i)*3+1,3) = AA.at<double>(1); B_eof.at<double>((i)*3+1,4) =  BB.at<double>(1); B_eof.at<double>((i)*3+1,5) = CC.at<double>(1);
            B_eof.at<double>((i)*3+2,2) = 1; B_eof.at<double>((i)*3+2,3) = AA.at<double>(2); B_eof.at<double>((i)*3+2,4) =  BB.at<double>(2); B_eof.at<double>((i)*3+2,5) = CC.at<double>(2);
        }

        for(int i = 0; i < _equcoor.rows; i++){
            L_norm.at<double>((i)*3+0) = X_obj.at<double>(i)-(R_array.at<double>(0,0)*X_start.at<double>(i)+R_array.at<double>(0,1)*Y_start.at<double>(i)+R_array.at<double>(0,2)*Z_start.at<double>(i))-X_delta;
            L_norm.at<double>((i)*3+1) = Y_obj.at<double>(i)-(R_array.at<double>(1,0)*X_start.at<double>(i)+R_array.at<double>(1,1)*Y_start.at<double>(i)+R_array.at<double>(1,2)*Z_start.at<double>(i))-Y_delta;
            L_norm.at<double>((i)*3+2) = Z_obj.at<double>(i)-(R_array.at<double>(2,0)*X_start.at<double>(i)+R_array.at<double>(2,1)*Y_start.at<double>(i)+R_array.at<double>(2,2)*Z_start.at<double>(i))-Z_delta;
         }


            cv::Mat NN_eof = B_eof.t() * B_eof;
            cv::Mat NL_norm = B_eof.t() * L_norm;

            cv::Mat NN_eof_inv;
            cv::invert(NN_eof, NN_eof_inv);
            cv::Mat X_val = NN_eof_inv * NL_norm;
            cv::norm(X_val, 2);
            X_delta = X_delta+X_val.at<double>(0);
            Y_delta = Y_delta+X_val.at<double>(1);
            Z_delta = Z_delta+X_val.at<double>(2);

            a = a+X_val.at<double>(3);
            b = b+X_val.at<double>(4);
            c = c+X_val.at<double>(5);

            if(loop_counter > 50000){
                std::cout<<"enough"<<std::endl;
                break;
            }
            else{
                if(loop_counter % 1000 == 0)
                    std::cout<<loop_counter<<"   "<<X_val<<std::endl;
                loop_counter++;
            }
    }

    //final R
    double R_norm = 1 + a*a + b*b + c*c;
    cv::Mat R_array = cv::Mat(3, 3, CV_64F);
    R_array.at<double>(0, 0) = 1 + a*a - b*b - c*c; R_array.at<double>(0, 1) = -2*c -2*a*b;         R_array.at<double>(0, 2) = -2*b+2*a*c;
    R_array.at<double>(0, 1) = 2*c - 2*b*a;         R_array.at<double>(1, 1) = 1 - a*a + b*b - c*c; R_array.at<double>(1, 2) = -2*a-2*b*c;
    R_array.at<double>(0, 2) = 2*b + 2*a*c;         R_array.at<double>(2, 1) = 2*a - 2*b*c;         R_array.at<double>(2, 2) = 1-a*a-b*b+c*c;

    R_array = R_array/R_norm;

    //based on Y, Roate by Y->X->Z， rotation angle is fi, omega, kapa
    double omega = -std::asin(R_array.at<double>(1,2));
    double comega = std::cos(omega);
    double fi = std::acos(R_array.at<double>(2, 2)/comega);
    double kapa = std::asin(R_array.at<double>(1,0)/comega);
    double Dpi=180.0/3.1415926;

    //
    cv::Mat Xobj_delt = cv::Mat::zeros(_equcoor.rows,1, CV_64F);
    cv::Mat Yobj_delt = cv::Mat::zeros(_equcoor.rows,1, CV_64F);
    cv::Mat Zobj_delt = cv::Mat::zeros(_equcoor.rows,1, CV_64F);

    cv::Mat XYZStart = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat Obj_cal = cv::Mat::zeros(3,1, CV_64F);
    cv::Mat XYZdelt = cv::Mat::zeros(3,1, CV_64F);

    XYZdelt.at<double>(0) = X_delta;
    XYZdelt.at<double>(1) = Y_delta;
    XYZdelt.at<double>(2) = Z_delta;

    cv::Mat LL;
    for(int i = 0; i < _equcoor.rows; i++){
        XYZStart.at<double>(0) = X_start.at<double>(i);
        XYZStart.at<double>(1) = Y_start.at<double>(i);
        XYZStart.at<double>(2) = Z_start.at<double>(i);

        Obj_cal = XYZdelt + R_array * XYZStart;
        Xobj_delt.at<double>(i) = Obj_cal.at<double>(0) - X_obj.at<double>(i);
        Yobj_delt.at<double>(i) = Obj_cal.at<double>(1) - Y_obj.at<double>(i);
        Zobj_delt.at<double>(i) = Obj_cal.at<double>(2) - Z_obj.at<double>(i);

        cv::Mat temp = cv::Mat::zeros(1, 3, CV_64F);
        temp.at<double>(0) = Xobj_delt.at<double>(i)/R.at<double>(i);
        temp.at<double>(1) = Yobj_delt.at<double>(i)/R.at<double>(i);
        temp.at<double>(2) = Zobj_delt.at<double>(i)/R.at<double>(i);

        LL.push_back(temp);
    }

    cv::Mat C2 = cv::abs(LL);

    double Xmax2, Xmin2;
    cv::minMaxLoc(C2.col(0), &Xmin2, &Xmax2);
    double Ymax2, Ymin2;
    cv::minMaxLoc(C2.col(1), &Ymin2, &Ymax2);
    double Zmax2, Zmin2;
    cv::minMaxLoc(C2.col(2), &Zmin2, &Zmax2);


    double Xave2 = cv::mean(C2.col(0))[0];
    double Yave2 = cv::mean(C2.col(1))[0];
    double Zave2 = cv::mean(C2.col(2))[0];

    _coorShift = cv::Mat::zeros(3, 1, CV_64F);
    _coorShift.at<float>(0) = X_delta;
    _coorShift.at<float>(1) = Y_delta;
    _coorShift.at<float>(2) = Z_delta;
    _transMatri = R_array.clone();
    _residualErr = cv::Mat::zeros(3, 1, CV_64F);
    _residualErr.at<float>(0) = Xave2;
    _residualErr.at<float>(1) = Yave2;
    _residualErr.at<float>(2) = Zave2;
}

//////////////////////////////////////////////////////////
///read a file from _data_path and return its data in _data
//////////////////////////////////////////////////////////
void read_data(std::string _data_path, cv::Mat& _data){
    std::ifstream ifstream;
    ifstream.open(_data_path.c_str());

    if(!ifstream.is_open()){
        std::cout<<"empty"<<std::endl;
        return;
    }

    std::string line;
    std::getline(ifstream, line);
    int counter = 0;
    ifstream >> counter ;
    std::getline(ifstream, line);

    cv::Mat data = cv::Mat::zeros(counter, 7, CV_64F);

    for(int i = 0; i < counter; i++){
        std::string line;
        std::getline(ifstream, line);
        std::stringstream ss;
        ss.str(line);
        for(int j = 0; j < 7; j++){
            double temp;
            ss >> temp;
            data.at<double>(i, j) = temp;
        }

    }

    _data = data;
}

int main(){
}