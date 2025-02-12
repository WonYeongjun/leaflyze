# leaflyze

초기 설정
1. 라즈베리파이와 컴퓨터를 동일한 와이파이로 연결한다
2. camera 폴더 안에 있는 button.py의 IP주소를 컴퓨터와 라즈베리가 연결되어있는 그 와이파이의 주소로 바꾼다.
3. 컴퓨터가 부팅하자마자 ssh 서버를 열도록 설정한다.
4. ArUco마커가 있는 직사각형의 가로와 세로의 길이를 실제로 재서 calibration_marker_detector.py의 correct_perspective함수의 width, height = 4200, 2970의 수치를 바꿔준다.
5. 컴퓨터에 C:/Users/UserK/Desktop/raw/로 폴더를 만들어놓는다.
6. 코드를 한 번 실행해보는데(방법은 밑에서 설명) 그때 잘 찾은 마커의 결과에서 scale을 확인해서 100미만이면 calmarker_new의 아래의 부분에서 fx와 fy를 줄이고, 반대로 100이상이면 키운다.
template_bgr = cv2.resize(
        template_bgr, (0, 0), fx=0.27, fy=0.27
    )  # 템플릿 사이즈 조절(초기 설정 필요)

실행방법
1. 라즈베리파이와 컴퓨터를 켠다.
2. 컴퓨터에서는 wait_for_image.py를 실행시킨다.
3. 라즈베리파이에서는 button.py를 실행시킨다.
4. 라즈베리파이에 있는 버튼을 누른다.
5. 기다리다가 컴퓨터에서 수많은 직사각형이 있는 이미지가 나올텐데, 그 중에 마커를 정확하게 가린 이미지의 인덱스를 4개 입력한다.

config.json 설명
{
  "username": "userk",//pc 사용자 이름

  "passward": "1234",//pc 비밀번호

  "ip_address": "192.168.1.100",//pc의 ip주소

  "raspi_file_path": "/home/userk/cal_img/raw/raw_img.jpg",//라즈베리파이 내부 사진 저장 위치

  "pc_file_path": "C:/Users/UserK/Desktop/raw/raw_img.jpg",//pc 내부 원본사진 저장 위치

  "pc_modified_file_path": "C:/Users/UserK/Desktop/fin/fin_img.jpg",//pc 내부 보정된 사진 저장 위치

  "ArUco_list": [12, 18, 27, 5],//ArUco 마커는 좌상단 우상단 우하단 좌하단 에 12, 18, 27, 5 순으로 배치(4x4크기의 마커)

  "ArUco_width": 4200, //(좌상단 마커의 우하단 꼭짓점 좌표와 우상단 마커의 좌하단 꼭짓점 좌표 사이의 거리)*n 직접 측정 후 기입

  "ArUco_height": 2970,//(좌상단 마커의 우하단 꼭짓점 좌표와 좌하단 마커의 우상단 꼭짓점 좌표 사이의 거리)*n 직접 측정 후 기입

  "matrix_coef": [
    [1.90296778e+03, 0.00000000e+00, 9.62538876e+02],
    [0.00000000e+00, 1.90259449e+03, 6.99587164e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
  ],//chessboard.py로 구한 mtx
  
  "dist_coeffs": [-0.00159553,  0.25363958, -0.00156397,  0.00185892, -0.41956662]//chessboard.py로 구한 dist

}