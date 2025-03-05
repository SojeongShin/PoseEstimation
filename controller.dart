import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:audioplayers/audioplayers.dart';
import 'package:camera/camera.dart';
import 'package:chewie/chewie.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:intl/intl.dart';
import 'package:tbox/global.dart';
import 'package:tbox/model.dart';
import 'package:tbox/module/bluetooth/bluetooth.dart';
import 'package:tbox/module/measurement/service.dart' as service;
import 'package:tbox/module/measurement/util.dart';
import 'package:video_player/video_player.dart';

abstract class PoseDetectionController extends GetxController {
  static Rect leftFootZone = isTablet.value
      ? const Rect.fromLTRB(200.0, 150.0, 265.0, 244.0)
      : const Rect.fromLTRB(568.0, 700.0, 768.0, 1020.0);
  static Rect rightFootZone = isTablet.value
      ? const Rect.fromLTRB(55.0, 150.0, 120.0, 244.0)
      : const Rect.fromLTRB(0.0, 700.0, 200.0, 1020.0);
  static Rect targetZone = isTablet.value
      ? const Rect.fromLTRB(120.0, 150.0, 200.0, 244.0)
      : const Rect.fromLTRB(200.0, 700.0, 568.0, 1020.0);

  CameraController? cameraController;
  final RxBool isCameraInit = false.obs;
  final RxBool _isFrameProcessing = false.obs;
  final posePaint = Rx<CustomPaint?>(null);

  final tboxService = Get.put(TboxService());
  final signalPlayer = AudioPlayer();
  final bgmPlayer = AudioPlayer();
  final stopwatch = Stopwatch();

  final currentState = 0.obs;
  final poseCount = 0.obs;
  final signalString = ''.obs;
  final timerString = ''.obs;
  final adviceString = ''.obs;
  final adviceState = 0.obs;
  final adviceTouch = false.obs;

  bool canCount = true;
  Timer? stopwatchPeriodical;
  Timer? signalPeriodical;

  final rotation = isTablet.value
      ? InputImageRotation.rotation0deg
      : (isCameraReversal.value
          ? InputImageRotation.rotation90deg
          : InputImageRotation.rotation270deg);

  final circularIndicatorPaint = Rx<CustomPaint?>(null);

  @override
  void onInit() {
    super.onInit();
    _initCamera();
    _initTbox();
    _initScreen();
  }

  @override
  void onClose() {
    cameraController?.dispose();
    tboxService.dispose();
    signalPlayer.dispose();
    bgmPlayer.dispose();
    stopwatchPeriodical?.cancel();
    signalPeriodical?.cancel();
    super.onClose();
  }

  void _initCamera();

  static Uint8List _concatenatePlanes(List<Plane> planes) {
    final int totalSize =
        planes.fold(0, (int size, Plane plane) => size + plane.bytes.length);
    final Uint8List bytes = Uint8List(totalSize);
    int offset = 0;

    for (Plane plane in planes) {
      final Uint8List planeBytes = plane.bytes;
      bytes.setRange(offset, offset + planeBytes.length, planeBytes);
      offset += planeBytes.length;
    }

    return bytes;
  }

  Future<InputImage> _convertCameraImageToInputImage(CameraImage image) async {
    final bytes = await compute(_concatenatePlanes, image.planes);

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: InputImageFormat.nv21,
        bytesPerRow: image.planes[0].bytesPerRow,
      ),
    );
  }

  void _initTbox() {
    tboxService.initTbox(onButton: () => _startProcess());
  }

  void _startProcess() {
    if (currentState.value > 2 ||
        tboxService.preButtonState[4] ||
        tboxService.preButtonState[5]) return;
    signalString.value = '';
    currentState.value = 3;

    signalPeriodical?.cancel();
    signalPeriodical =
        Timer.periodic(const Duration(seconds: 1), (timer) async {
      if (timer.tick <= 3) {
        await signalPlayer.play(AssetSource(
            "lib/module/measurement/assets/count_down_${4 - timer.tick}.mp3"));
        signalString.value = '${4 - timer.tick}';
      } else {
        timer.cancel();
        stopwatch.start();
        currentState.value = 4;
        await playBgm();
      }
    });
  }

  _initScreen() {
    signalPeriodical = Timer(const Duration(seconds: 1), () async {
      currentState.value = 1;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/ready.mp3"));
      signalPeriodical = Timer(const Duration(seconds: 1), () {
        currentState.value = 2;
      });
    });
  }

  bool allAnglesReset(double leftAngle, double rightAngle) {
    return false;
  }

  Future<void> playBgm() async {}
}

class MotionSquatController extends PoseDetectionController {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(),
  );

  static const int processTime = 40000;

  static const double squatDepthThreshold = -15.0;
  static const double squatAngleThreshold = 45.0;
  static const double backAngleThreshold = 50.0;
  static const double resetAngleThreshold = 10.0;
  static const double symmetryThreshold = 20.0;

  final leftAngle = 0.0.obs;
  final rightAngle = 0.0.obs;

  final isFacingForward = false.obs;
  final isBackStraight = false.obs;
  final isKneesBent = false.obs;

  bool isInitialPositionSet = false;

  DateTime lastAdviceTime = DateTime.now();

  @override
  void _initCamera() async {
    final List<CameraDescription> cameras = await availableCameras();
    final CameraDescription camera = isTablet.value
        ? cameras.firstWhere(
            (element) => element.lensDirection == CameraLensDirection.front)
        : cameras[0];

    cameraController = isTablet.value
        ? CameraController(
            camera,
            ResolutionPreset.low,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          )
        : CameraController(
            camera,
            ResolutionPreset.max,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          );

    cameraController!.initialize().then((value) {
      isCameraInit.value = true;
      cameraController!.startImageStream(_imageAnalysis);
    });
  }

  void _imageAnalysis(CameraImage cameraImage) async {
    if (_isFrameProcessing.value) return;

    _isFrameProcessing.value = true;
    final inputImage = await _convertCameraImageToInputImage(cameraImage);
    final poses = await _poseDetector.processImage(inputImage);
    _processPoses(poses, cameraImage);
    _isFrameProcessing.value = false;
  }

  @override
  Future<void> playBgm() async {
    await bgmPlayer.play(
        AssetSource('lib/module/measurement/assets/measurement_01_bgm.mp3'));
  }

  void _processPoses(List<Pose> poses, CameraImage image) async {
    if (poses.isEmpty) return;

    final pose = poses.first;
    final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    final leftKnee = pose.landmarks[PoseLandmarkType.leftKnee];
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightHip = pose.landmarks[PoseLandmarkType.rightHip];
    final rightKnee = pose.landmarks[PoseLandmarkType.rightKnee];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];

    if (leftHip != null &&
        leftKnee != null &&
        leftAnkle != null &&
        rightHip != null &&
        rightKnee != null &&
        rightAnkle != null) {
      leftAngle.value = calculateAngle(leftHip, leftKnee, leftAnkle);
      rightAngle.value = calculateAngle(rightHip, rightKnee, rightAnkle);

      if (canCount && currentState.value == 4) {
        if (isValidSquat(pose, leftAngle.value, rightAngle.value)) {
          if (tboxService.preButtonState[4] || tboxService.preButtonState[5]) {
            poseCount.value++;
            canCount = false;
            adviceTouch.value = false;
            await signalPlayer.play(AssetSource(
                "lib/module/measurement/assets/count_${poseCount.value}.mp3"));
          } else if ((!tboxService.preButtonState[4] &&
                  !tboxService.preButtonState[5]) &&
              adviceState.value == 0) {
            adviceTouch.value = true;
          }
        } else if (canCount &&
            !allAnglesReset(leftAngle.value, rightAngle.value) &&
            adviceState.value == 0 &&
            !adviceTouch.value) {
          if (DateTime.now().difference(lastAdviceTime).inSeconds > 1) {
            await _provideAdvice();
            lastAdviceTime = DateTime.now();
          }
        }
      }

      if (!canCount && allAnglesReset(leftAngle.value, rightAngle.value)) {
        canCount = true;
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/whistle_short.mp3"));
      } else if (canCount &&
          allAnglesReset(leftAngle.value, rightAngle.value) &&
          adviceTouch.value) {
        adviceString.value = '티박스를 터치해주세요';
        adviceTouch.value = false;
        await signalPlayer
            .play(AssetSource("lib/module/measurement/assets/touch.mp3"));
      }

      posePaint.value = CustomPaint(
        painter: PosePainter(
          poses,
          Size(image.width.toDouble(), image.height.toDouble()),
          isValidSquat(pose, leftAngle.value, rightAngle.value),
        ),
      );
    }
  }

  Future<void> _provideAdvice() async {
    if (!isFacingForward.value) {
      adviceState.value = 1;
      adviceString.value = '정면을 봐주세요';
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/shoulder.mp3"));
    } else if (!isBackStraight.value) {
      adviceState.value = 2;
      adviceString.value = '허리를 펴주세요';
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/back_angle.mp3"));
    }
  }

  bool isValidSquat(Pose pose, double leftAngle, double rightAngle) {
    final Map<String, bool> validationResults = {
      'isFacingForward': _isFacingForward(pose),
      'isBackStraight': _isBackStraight(pose),
      'isKneesBent': _isKneesBent(),
    };

    isFacingForward.value = validationResults['isFacingForward']!;
    isBackStraight.value = validationResults['isBackStraight']!;
    isKneesBent.value = validationResults['isKneesBent']!;

    switch (adviceState.value) {
      case 1:
        if (isFacingForward.value) {
          adviceState.value = 0;
          adviceString.value = '';
        }
        break;
      case 2:
        if (isBackStraight.value) {
          adviceState.value = 0;
          adviceString.value = '';
        }
        break;
    }

    return validationResults.values.every((result) => result);
  }

  bool _isFacingForward(Pose pose) {
    final nose = pose.landmarks[PoseLandmarkType.nose];
    final leftEye = pose.landmarks[PoseLandmarkType.leftEye];
    final rightEye = pose.landmarks[PoseLandmarkType.rightEye];

    if (nose == null || leftEye == null || rightEye == null) return false;

    return (nose.x - leftEye.x).abs() < (nose.x - rightEye.x).abs() * 1.5 &&
        (nose.x - rightEye.x).abs() < (nose.x - leftEye.x).abs() * 1.5;
  }

  bool _isBackStraight(Pose pose) {
    final leftShoulder = pose.landmarks[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose.landmarks[PoseLandmarkType.rightShoulder];
    final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    final rightHip = pose.landmarks[PoseLandmarkType.rightHip];

    if (leftShoulder == null ||
        rightShoulder == null ||
        leftHip == null ||
        rightHip == null) return false;

    // 양쪽 어깨와 엉덩이의 x축 차이도 확인 (허리의 좌우 비대칭 방지)
    bool isLeftSideStraight =
        (leftShoulder.x - leftHip.x).abs() < backAngleThreshold;
    bool isRightSideStraight =
        (rightShoulder.x - rightHip.x).abs() < backAngleThreshold;

    return isLeftSideStraight && isRightSideStraight;
  }

  bool _isKneesBent() {
    // 무릎 각도가 일정 임계값 이상으로 굽혀졌는지 확인합니다.
    final bool kneeAngleValid =
        leftAngle >= squatAngleThreshold && rightAngle >= squatAngleThreshold;

    return kneeAngleValid;
  }

  @override
  _initScreen() async {
    signalPeriodical = Timer(const Duration(seconds: 1), () async {
      currentState.value = 1;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/ready.mp3"));
      signalPeriodical = Timer(const Duration(seconds: 1), () {
        currentState.value = 2;
      });
    });

    stopwatchPeriodical =
        Timer.periodic(const Duration(milliseconds: 33), (Timer t) async {
      if (currentState.value == 4 &&
          stopwatch.elapsedMilliseconds >= processTime) {
        currentState.value = 5;
        stopwatch.stop();
        await bgmPlayer.stop();
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/whistle_done.mp3"));

        final grade = await _calculateGrade(poseCount.value);
        signalPeriodical = Timer(const Duration(seconds: 1), () async {
          if (isLocal.value == false && profileId.value > 0) {
            final trackingId = (await service.addTrackingOrganization(
                400, 400, 100))['tracking_id'];
            service.addTrackingDataOrganization(
                trackingId, 110, stopwatch.elapsed.inSeconds);
            service.addTrackingDataOrganization(
                trackingId, 200, poseCount.value);
            service.addTrackingDataOrganization(trackingId, 201, grade);
          }
          Get.offNamed('/mission/measurement/result',
              arguments: MeasurementResultData(
                resultNum: poseCount.value,
                grade: grade,
                testType: 1,
              ));
        });
      }
      if (stopwatch.elapsedMilliseconds < processTime) {
        timerString.value =
            '00:${NumberFormat('00').format(((processTime / 1000) - (stopwatch.elapsedMilliseconds / 1000) % 60).floor())}:${(NumberFormat('00').format((100 - stopwatch.elapsedMilliseconds % 1000 ~/ 10) % 100))}';
      } else {
        timerString.value = '00:00:00';
      }
    });
  }

  Future<int> _calculateGrade(int value) async {
    String jsonString = await rootBundle
        .loadString('lib/module/measurement/assets/measurement_grade.json');
    final jsonResponse = await json.decode(jsonString);

    if (value >= jsonResponse[0][userMetricsType]["1"]) return 1;
    if (value >= jsonResponse[0][userMetricsType]["2"]) return 2;
    if (value >= jsonResponse[0][userMetricsType]["3"]) return 3;
    if (value >= jsonResponse[0][userMetricsType]["4"]) return 4;
    return 5;
  }

  @override
  bool allAnglesReset(double leftAngle, double rightAngle) {
    return leftAngle < resetAngleThreshold && rightAngle < resetAngleThreshold;
  }
}

class MotionForwardBendController extends PoseDetectionController {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(),
  );

  static const double sizePerPixelKiosk = 0.200662;
  static const double sizePerPixelTablet = 0.092736;

  // static const double kneeAngleThreshold = 30.0;
  // static const double footDistanceThreshold = 100.0;

  final kneeAngle = 0.0.obs;
  final handFootDistance = 0.0.obs;
  final footDistance = 0.0.obs;

  final isKneesStraight = false.obs;
  final isFootAlign = false.obs;

  Timer? poseHoldTimer;
  bool isHoldingPose = false;
  final holdTime = 0.0.obs;
  final holdSignalState = 0.obs;
  DateTime lastHoldCheckTime = DateTime.now();

  DateTime lastAdviceTime = DateTime.now();

  @override
  void _initCamera() async {
    final List<CameraDescription> cameras = await availableCameras();
    final CameraDescription camera = isTablet.value
        ? cameras.firstWhere(
            (element) => element.lensDirection == CameraLensDirection.front)
        : cameras[0];

    cameraController = CameraController(
      camera,
      ResolutionPreset.max,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21,
    );

    cameraController!.initialize().then((value) {
      isCameraInit.value = true;
      cameraController!.startImageStream(_imageAnalysis);
    });
  }

  void _imageAnalysis(CameraImage cameraImage) async {
    if (_isFrameProcessing.value) return;

    _isFrameProcessing.value = true;
    final inputImage = await _convertCameraImageToInputImage(cameraImage);
    final poses = await _poseDetector.processImage(inputImage);
    _processPoses(poses, cameraImage);
    _isFrameProcessing.value = false;
  }

  @override
  Future<void> playBgm() async {
    await bgmPlayer
        .play(AssetSource('lib/module/measurement/assets/measurement_02_bgm'));
  }

  void _processPoses(List<Pose> poses, CameraImage image) async {
    if (poses.isEmpty) return;

    final pose = poses.first;
    final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    final rightHip = pose.landmarks[PoseLandmarkType.rightHip];
    final leftKnee = pose.landmarks[PoseLandmarkType.leftKnee];
    final rightKnee = pose.landmarks[PoseLandmarkType.rightKnee];
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];
    final leftIndex = pose.landmarks[PoseLandmarkType.leftIndex];
    final rightIndex = pose.landmarks[PoseLandmarkType.rightIndex];
    final leftFootIndex = pose.landmarks[PoseLandmarkType.leftFootIndex];
    final rightFootIndex = pose.landmarks[PoseLandmarkType.rightFootIndex];

    if (leftHip != null &&
        rightHip != null &&
        leftKnee != null &&
        rightKnee != null &&
        leftAnkle != null &&
        rightAnkle != null &&
        leftIndex != null &&
        rightIndex != null &&
        leftFootIndex != null &&
        rightFootIndex != null &&
        currentState.value < 5) {
      kneeAngle.value = min(calculateAngle(leftHip, leftKnee, leftAnkle),
          calculateAngle(rightHip, rightKnee, rightAnkle));

      footDistance.value = (leftAnkle.x - rightAnkle.x).abs();
      handFootDistance.value =
          (((leftIndex.y > rightIndex.y ? leftIndex.y : rightIndex.y) -
                      ((leftFootIndex.y + rightFootIndex.y) / 2.0)) *
                  (isTablet.value ? sizePerPixelTablet : sizePerPixelKiosk)) +
              9.0;

      if (currentState.value == 4) {
        if (isValidForwardBend(pose, kneeAngle.value, footDistance.value) &&
            (tboxService.preButtonState[4] || tboxService.preButtonState[5])) {
          _startPoseHold();
        } else {
          _resetPoseHold();
          if (adviceState.value == 0 && !isKneesStraight.value) {
            if (DateTime.now().difference(lastAdviceTime).inSeconds > 1) {
              await _giveAdvice(
                  '무릎을 펴주세요', 'lib/module/measurement/assets/knee_angle.mp3');
              lastAdviceTime = DateTime.now();
            }
          }
        }
      }

      posePaint.value = CustomPaint(
        painter: PosePainter(
          poses,
          Size(image.width.toDouble(), image.height.toDouble()),
          isValidForwardBend(pose, kneeAngle.value, footDistance.value),
        ),
      );

      circularIndicatorPaint.value = CustomPaint(
        painter: CircularIndicatorPainter(holdTime.value),
      );
    }
  }

  bool isValidForwardBend(Pose pose, double kneeAngle, double footDistance) {
    // isKneesStraight.value = kneeAngle < kneeAngleThreshold;
    // isFootAlign.value = footDistance < footDistanceThreshold;

    // final leftHip = pose.landmarks[PoseLandmarkType.leftHip];
    // final rightHip = pose.landmarks[PoseLandmarkType.rightHip];
    // final leftKnee = pose.landmarks[PoseLandmarkType.leftKnee];
    // final rightKnee = pose.landmarks[PoseLandmarkType.rightKnee];
    // if (leftHip == null ||
    //     rightHip == null ||
    //     leftKnee == null ||
    //     rightKnee == null) {
    //   return false;
    // }

    // if (adviceState.value == 1 && isKneesStraight.value) {
    //   _clearAdvice();
    // } else if (adviceState.value == 2 && isFootAlign.value) {
    //   _clearAdvice();
    // }

    // return isKneesStraight.value && isFootAlign.value;
    return true;
  }

  void _startPoseHold() {
    if (!isHoldingPose) {
      isHoldingPose = true;
      holdTime.value = 0;
      lastHoldCheckTime = DateTime.now();
      poseHoldTimer =
          Timer.periodic(const Duration(milliseconds: 100), (timer) async {
        holdTime.value += 0.1;
        await _playHoldSignal();
        if (holdTime.value >= 3.0) {
          _completePoseHold(timer);
        }
      });
    }
  }

  Future<void> _playHoldSignal() async {
    if (holdTime.value >= 0.0 && holdSignalState.value == 0) {
      holdSignalState.value = 1;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/count_down_3.mp3"));
    } else if (holdTime.value >= 1.0 && holdSignalState.value == 1) {
      holdSignalState.value = 2;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/count_down_2.mp3"));
    } else if (holdTime.value >= 2.0 && holdSignalState.value == 2) {
      holdSignalState.value = 3;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/count_down_1.mp3"));
    }
  }

  Future<void> _completePoseHold(Timer timer) async {
    currentState.value = 5;
    timer.cancel();
    await bgmPlayer.stop();
    await signalPlayer
        .play(AssetSource("lib/module/measurement/assets/whistle_done.mp3"));

    final grade = await _calculateGrade(handFootDistance.value);
    signalPeriodical = Timer(const Duration(seconds: 1), () async {
      if (isLocal.value == false && profileId.value > 0) {
        final trackingId = (await service.addTrackingOrganization(
            400, 400, 200))['tracking_id'];
        service.addTrackingDataOrganization(
            trackingId, 210, (handFootDistance.value * 10 + 10000).toInt());
        service.addTrackingDataOrganization(trackingId, 211, grade);
      }
      Get.offNamed('/mission/measurement/result',
          arguments: MeasurementResultData(
            resultNum: handFootDistance.value,
            grade: grade,
            testType: 2,
          ));
    });
    isHoldingPose = false;
  }

  Future<int> _calculateGrade(double value) async {
    String jsonString = await rootBundle
        .loadString('lib/module/measurement/assets/measurement_grade.json');
    final jsonResponse = await json.decode(jsonString);

    if (value >= jsonResponse[1][userMetricsType]["1"]) return 1;
    if (value >= jsonResponse[1][userMetricsType]["2"]) return 2;
    if (value >= jsonResponse[1][userMetricsType]["3"]) return 3;
    if (value >= jsonResponse[1][userMetricsType]["4"]) return 4;
    return 5;
  }

  void _resetPoseHold() {
    isHoldingPose = false;
    holdTime.value = 0;
    holdSignalState.value = 0;
    poseHoldTimer?.cancel();
  }

  Future<void> _giveAdvice(String message, String assetPath) async {
    adviceState.value = 1;
    adviceString.value = message;
    await signalPlayer.play(AssetSource(assetPath));
  }

  void _clearAdvice() {
    adviceState.value = 0;
    adviceString.value = '';
  }

  @override
  void onClose() {
    poseHoldTimer?.cancel();
    super.onClose();
  }
}

class MotionSideStepController extends PoseDetectionController {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(),
  );

  static const int processTime = 20000;

  static const double groundThreshold = 20.0;

  final leftFootZoneState = FootZone.none.obs;
  final rightFootZoneState = FootZone.none.obs;

  int currentStep = 0;
  bool isFirstStep = true;

  @override
  void _initCamera() async {
    final List<CameraDescription> cameras = await availableCameras();
    final CameraDescription camera = isTablet.value
        ? cameras.firstWhere(
            (element) => element.lensDirection == CameraLensDirection.front)
        : cameras[0];

    cameraController = isTablet.value
        ? CameraController(
            camera,
            ResolutionPreset.low,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          )
        : CameraController(
            camera,
            ResolutionPreset.max,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          );

    cameraController!.initialize().then((value) {
      isCameraInit.value = true;
      cameraController!.startImageStream(_imageAnalysis);
    });
  }

  void _imageAnalysis(CameraImage cameraImage) async {
    if (_isFrameProcessing.value) return;

    _isFrameProcessing.value = true;
    final inputImage = await _convertCameraImageToInputImage(cameraImage);
    final poses = await _poseDetector.processImage(inputImage);
    _processPoses(poses, cameraImage);
    _isFrameProcessing.value = false;
  }

  @override
  Future<void> playBgm() async {
    await bgmPlayer.play(
        AssetSource('lib/module/measurement/assets/measurement_03_bgm.mp3'));
  }

  void _processPoses(List<Pose> poses, CameraImage image) async {
    if (poses.isEmpty) return;

    final pose = poses.first;
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];

    if (leftAnkle == null || rightAnkle == null) return;

    leftFootZoneState.value = _determineFootZone(leftAnkle);
    rightFootZoneState.value = _determineFootZone(rightAnkle);

    if (_isSideStepComplete() &&
        currentState.value == 4 &&
        _isFeetOnGround(pose)) {
      poseCount.value++;
      currentStep++;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/whistle_short.mp3"));
      _resetStep();
    }

    posePaint.value = CustomPaint(
      painter: PosePainter(
        poses,
        Size(image.width.toDouble(), image.height.toDouble()),
        true,
      ),
    );

    // posePaint.value = CustomPaint(
    //   painter: PosePainter(poses, imageSize, true,
    //       leftFootZone: PoseDetectionController.leftFootZone,
    //       rightFootZone: PoseDetectionController.rightFootZone,
    //       targetZone: PoseDetectionController.targetZone),
  }

  FootZone _determineFootZone(PoseLandmark ankle) {
    final Offset ankleOffset = Offset(ankle.x, ankle.y);

    if (PoseDetectionController.leftFootZone.contains(ankleOffset)) {
      return FootZone.left;
    } else if (PoseDetectionController.rightFootZone.contains(ankleOffset)) {
      return FootZone.right;
    } else if (PoseDetectionController.targetZone.contains(ankleOffset)) {
      return FootZone.target;
    } else {
      return FootZone.none;
    }
  }

  bool _isSideStepComplete() {
    if (isFirstStep) {
      if ((leftFootZoneState.value == FootZone.left &&
              rightFootZoneState.value != FootZone.right &&
              tboxService.preButtonState[4] &&
              !tboxService.preButtonState[5]) ||
          (rightFootZoneState.value == FootZone.right &&
              leftFootZoneState.value != FootZone.left &&
              !tboxService.preButtonState[4] &&
              tboxService.preButtonState[5])) {
        isFirstStep = false;
        return true;
      }
    } else {
      switch (currentStep % 4) {
        case 0:
          return leftFootZoneState.value == FootZone.left &&
              rightFootZoneState.value != FootZone.right &&
              tboxService.preButtonState[4] &&
              !tboxService.preButtonState[5];
        case 1:
        case 3:
          return leftFootZoneState.value == FootZone.target &&
              rightFootZoneState.value == FootZone.target &&
              !tboxService.preButtonState[4] &&
              !tboxService.preButtonState[5];
        case 2:
          return leftFootZoneState.value != FootZone.left &&
              rightFootZoneState.value == FootZone.right &&
              !tboxService.preButtonState[4] &&
              tboxService.preButtonState[5];
      }
    }
    return false;
  }

  bool _isFeetOnGround(Pose pose) {
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];

    if (leftAnkle == null || rightAnkle == null) return false;

    return leftAnkle.y >= groundThreshold && rightAnkle.y >= groundThreshold;
  }

  void _resetStep() {
    leftFootZoneState.value = FootZone.none;
    rightFootZoneState.value = FootZone.none;
    if (currentStep == 4) {
      currentStep = 0;
    }
  }

  @override
  _initScreen() async {
    signalPeriodical = Timer(const Duration(seconds: 1), () async {
      currentState.value = 1;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/ready.mp3"));
      signalPeriodical = Timer(const Duration(seconds: 1), () {
        currentState.value = 2;
      });
    });

    stopwatchPeriodical =
        Timer.periodic(const Duration(milliseconds: 33), (Timer t) async {
      if (currentState.value == 4 &&
          stopwatch.elapsedMilliseconds >= processTime) {
        currentState.value = 5;
        stopwatch.stop();
        await bgmPlayer.stop();
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/whistle_done.mp3"));

        final grade = await _calculateGrade(poseCount.value);
        signalPeriodical = Timer(const Duration(seconds: 1), () async {
          if (isLocal.value == false && profileId.value > 0) {
            final trackingId = (await service.addTrackingOrganization(
                400, 400, 300))['tracking_id'];
            service.addTrackingDataOrganization(
                trackingId, 110, stopwatch.elapsed.inSeconds);
            service.addTrackingDataOrganization(
                trackingId, 220, poseCount.value);
            service.addTrackingDataOrganization(trackingId, 221, grade);
          }
          Get.offNamed('/mission/measurement/result',
              arguments: MeasurementResultData(
                resultNum: poseCount.value,
                grade: grade,
                testType: 3,
              ));
        });
      }
      if (stopwatch.elapsedMilliseconds < processTime) {
        timerString.value =
            '00:${NumberFormat('00').format(((processTime / 1000) - (stopwatch.elapsedMilliseconds / 1000) % 60).floor())}:${(NumberFormat('00').format((100 - stopwatch.elapsedMilliseconds % 1000 ~/ 10) % 100))}';
      } else {
        timerString.value = '00:00:00';
      }
    });
  }

  Future<int> _calculateGrade(int value) async {
    String jsonString = await rootBundle
        .loadString('lib/module/measurement/assets/measurement_grade.json');
    final jsonResponse = await json.decode(jsonString);

    if (value >= jsonResponse[2][userMetricsType]["1"]) return 1;
    if (value >= jsonResponse[2][userMetricsType]["2"]) return 2;
    if (value >= jsonResponse[2][userMetricsType]["3"]) return 3;
    if (value >= jsonResponse[2][userMetricsType]["4"]) return 4;
    return 5;
  }
}

class MotionSoccerRunController extends PoseDetectionController {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(),
  );

  static const int processTime = 60000;

  final objPosition = 0.obs;
  final objImage = 'lib/module/measurement/assets/touch_soccer_obj_i.png';
  final objColor = 0xFFFF00;

  List<bool> nextbuttonState = [false, false, false, false, false, false];

  final leftFootZoneState = FootZone.none.obs;
  final rightFootZoneState = FootZone.none.obs;

  final currentStep = 0.obs;

  @override
  void _initCamera() async {
    final List<CameraDescription> cameras = await availableCameras();
    final CameraDescription camera = isTablet.value
        ? cameras.firstWhere(
            (element) => element.lensDirection == CameraLensDirection.front)
        : cameras[0];

    cameraController = isTablet.value
        ? CameraController(
            camera,
            ResolutionPreset.low,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          )
        : CameraController(
            camera,
            ResolutionPreset.max,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          );

    cameraController!.initialize().then((value) {
      isCameraInit.value = true;
      cameraController!.startImageStream(_imageAnalysis);
    });
  }

  void _imageAnalysis(CameraImage cameraImage) async {
    if (_isFrameProcessing.value) return;

    _isFrameProcessing.value = true;
    final inputImage = await _convertCameraImageToInputImage(cameraImage);
    final poses = await _poseDetector.processImage(inputImage);
    _processPoses(poses, cameraImage);
    _isFrameProcessing.value = false;
  }

  @override
  void _initTbox() {
    tboxService.initTbox(onButton: () => _processStep());
  }

  void _processStep() {
    if (currentState.value == 2 &&
        !tboxService.preButtonState[4] &&
        !tboxService.preButtonState[5]) {
      _startProcess();
    } else if (currentState.value == 4) {
      if (currentStep.value == 0 &&
          tboxService.preButtonState[4] &&
          !tboxService.preButtonState[5]) {
        _advanceStep([false, false, false, false, false, true]);
      } else if (currentStep.value == 0 &&
          !tboxService.preButtonState[4] &&
          tboxService.preButtonState[5]) {
        _advanceStep([false, false, false, false, true, false]);
      } else if ((currentStep.value == 1 || currentStep.value == 2) &&
          listEquals(nextbuttonState, tboxService.preButtonState)) {
        _advanceStep([
          false,
          false,
          false,
          false,
          !nextbuttonState[4],
          !nextbuttonState[5]
        ]);
      } else if (currentStep.value == 3 &&
          listEquals(nextbuttonState, tboxService.preButtonState)) {
        currentStep.value++;
      } else if (currentStep.value == 4 &&
          !tboxService.preButtonState[4] &&
          !tboxService.preButtonState[5]) {
        currentState.value++;
        _setProcess();
      }
    }
  }

  void _advanceStep(List<bool> nextState) {
    currentStep.value++;
    nextbuttonState = nextState;
  }

  @override
  Future<void> playBgm() async {
    await bgmPlayer.play(
        AssetSource('lib/module/measurement/assets/measurement_04_bgm.mp3'));
  }

  void _processPoses(List<Pose> poses, CameraImage image) async {
    if (poses.isEmpty) return;

    final pose = poses.first;
    final leftAnkle = pose.landmarks[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose.landmarks[PoseLandmarkType.rightAnkle];

    if (leftAnkle == null || rightAnkle == null) return;

    leftFootZoneState.value = _determineFootZone(leftAnkle);
    rightFootZoneState.value = _determineFootZone(rightAnkle);

    if (_isSideStepComplete() && currentState.value == 5) {
      _resetProcess();
      poseCount.value++;
      currentState.value = 4;
      currentStep.value = 0;
      await signalPlayer.play(AssetSource(
          "lib/module/measurement/assets/count_${poseCount.value}.mp3"));
    }

    posePaint.value = CustomPaint(
      painter: PosePainter(
        poses,
        Size(image.width.toDouble(), image.height.toDouble()),
        true,
      ),
    );

    // posePaint.value = CustomPaint(
    //   painter: PosePainter(poses, imageSize, true,
    //       leftFootZone: PoseDetectionController.leftFootZone,
    //       rightFootZone: PoseDetectionController.rightFootZone,
    //       targetZone: PoseDetectionController.targetZone),
  }

  FootZone _determineFootZone(PoseLandmark ankle) {
    final Offset ankleOffset = Offset(ankle.x, ankle.y);

    if (PoseDetectionController.leftFootZone.contains(ankleOffset)) {
      return FootZone.left;
    } else if (PoseDetectionController.rightFootZone.contains(ankleOffset)) {
      return FootZone.right;
    } else if (PoseDetectionController.targetZone.contains(ankleOffset)) {
      return FootZone.target;
    } else {
      return FootZone.none;
    }
  }

  bool _isSideStepComplete() {
    return currentState.value == 5 &&
        ((objPosition.value == 1 &&
                leftFootZoneState.value == FootZone.left &&
                rightFootZoneState.value != FootZone.right) ||
            (objPosition.value == 2 &&
                leftFootZoneState.value != FootZone.left &&
                rightFootZoneState.value == FootZone.right));
  }

  void _resetProcess() {
    objPosition.value = 0;
    sendMessage(
        [packetHeader, 0x06, 0x10, 0xff, 0x00, 0x00, 0x00, 0x00, packetFooter]);
  }

  void _setProcess() async {
    objPosition.value = Random().nextInt(2) + 1;
    await signalPlayer
        .play(AssetSource("lib/module/measurement/assets/whistle_short.mp3"));
    final selectedColor = Color(objColor);
    final r = selectedColor.red;
    final g = selectedColor.green;
    final b = selectedColor.blue;
    for (int i = 0; i < 4; i++) {
      sendMessage([packetHeader, 0x06, 0x10, i, 0x01, r, g, b, packetFooter]);
    }
  }

  @override
  _initScreen() async {
    signalPeriodical = Timer(const Duration(seconds: 1), () async {
      currentState.value = 1;
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/ready.mp3"));
      signalPeriodical = Timer(const Duration(seconds: 1), () {
        currentState.value = 2;
      });
    });

    stopwatchPeriodical =
        Timer.periodic(const Duration(milliseconds: 33), (Timer t) async {
      if ([4, 5].contains(currentState.value) &&
          stopwatch.elapsedMilliseconds >= processTime) {
        currentState.value = 6;
        stopwatch.stop();
        await bgmPlayer.stop();
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/whistle_done.mp3"));

        final grade = await _calculateGrade(poseCount.value);
        signalPeriodical = Timer(const Duration(seconds: 1), () async {
          if (isLocal.value == false && profileId.value > 0) {
            final trackingId = (await service.addTrackingOrganization(
                400, 400, 400))['tracking_id'];
            service.addTrackingDataOrganization(
                trackingId, 110, stopwatch.elapsed.inSeconds);
            service.addTrackingDataOrganization(
                trackingId, 230, poseCount.value);
            service.addTrackingDataOrganization(trackingId, 231, grade);
          }
          Get.offNamed('/mission/measurement/result',
              arguments: MeasurementResultData(
                resultNum: (poseCount.value * 4) + currentStep.value,
                grade: grade,
                testType: 4,
              ));
        });
      }
      if (stopwatch.elapsedMilliseconds < processTime) {
        timerString.value =
            '00:${NumberFormat('00').format(((processTime / 1000) - (stopwatch.elapsedMilliseconds / 1000) % 60).floor())}:${(NumberFormat('00').format((100 - stopwatch.elapsedMilliseconds % 1000 ~/ 10) % 100))}';
      } else {
        timerString.value = '00:00:00';
      }
    });
  }

  Future<int> _calculateGrade(int value) async {
    String jsonString = await rootBundle
        .loadString('lib/module/measurement/assets/measurement_grade.json');
    final jsonResponse = await json.decode(jsonString);

    if (value >= jsonResponse[3][userMetricsType]["1"]) return 1;
    if (value >= jsonResponse[3][userMetricsType]["2"]) return 2;
    if (value >= jsonResponse[3][userMetricsType]["3"]) return 3;
    if (value >= jsonResponse[3][userMetricsType]["4"]) return 4;
    return 5;
  }
}

class MotionHeightController extends PoseDetectionController {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(model: PoseDetectionModel.accurate),
  );

  static const double heightPerPixelKiosk = 0.276409;
  static const double heightPerPixelTablet = 0.125038;
  static const int footPositionMinKiosk = 830;
  static const int footPositionMaxKiosk = 850;
  static const int footPositionMinTablet = 2090;
  static const int footPositionMaxTablet = 2150;

  // final heelAngleThreshold = 15.0; // 발 뒤꿈치가 들렸다고 판단할 각도 임계값
  final bodyHeight = 0.0.obs;
  final bodyHeightPixel = 0.0.obs;
  final footPosition = 0.0.obs;
  final eyePosition = 0.0.obs;
  final leftHeelAngleDebug = 0.0.obs;
  final rightHeelAngleDebug = 0.0.obs;
  final isHeelGround = false.obs;

  final List<double> heightMeasurements = [];
  Timer? holdTimer;
  bool isHoldingPosition = false;
  final holdTime = 0.0.obs; // 현재 유지된 시간
  final holdSignalState = 0.obs; // 음성 신호 상태

  @override
  void _initCamera() async {
    final List<CameraDescription> cameras = await availableCameras();
    final CameraDescription camera = isTablet.value
        ? cameras.firstWhere(
            (element) => element.lensDirection == CameraLensDirection.front)
        : cameras[0];

    cameraController = CameraController(
      camera,
      ResolutionPreset.max,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21,
    );

    cameraController!.initialize().then((value) {
      isCameraInit.value = true;
      cameraController!.startImageStream(_imageAnalysis);
    });
  }

  void _imageAnalysis(CameraImage cameraImage) async {
    if (_isFrameProcessing.value) return;

    _isFrameProcessing.value = true;
    final inputImage = await _convertCameraImageToInputImage(cameraImage);
    final poses = await _poseDetector.processImage(inputImage);
    _processPoses(poses, cameraImage);
    _isFrameProcessing.value = false;
  }

  @override
  Future<void> playBgm() async {
    await bgmPlayer.play(
        AssetSource('lib/module/measurement/assets/measurement_02_bgm.mp3'));
  }

  void _processPoses(List<Pose> poses, CameraImage image) async {
    if (poses.isEmpty) return;

    final pose = poses.first;
    final leftHeel = pose.landmarks[PoseLandmarkType.leftHeel];
    final rightHeel = pose.landmarks[PoseLandmarkType.rightHeel];
    final leftFootIndex = pose.landmarks[PoseLandmarkType.leftFootIndex];
    final rightFootIndex = pose.landmarks[PoseLandmarkType.rightFootIndex];
    final leftEye = pose.landmarks[PoseLandmarkType.leftEye];
    final rightEye = pose.landmarks[PoseLandmarkType.rightEye];

    if (leftHeel == null ||
        rightHeel == null ||
        leftFootIndex == null ||
        rightFootIndex == null ||
        leftEye == null ||
        rightEye == null ||
        currentState.value > 4) return;

    footPosition.value = (leftFootIndex.y + rightFootIndex.y) / 2.0;
    eyePosition.value = (leftEye.y + rightEye.y) / 2.0;
    bodyHeightPixel.value = footPosition.value - eyePosition.value;

    isTablet.value
        ? bodyHeight.value = bodyHeightPixel.value * heightPerPixelTablet + 11.0
        : bodyHeight.value = bodyHeightPixel.value * heightPerPixelKiosk + 11.0;

    // isTablet.value
    //     ? bodyHeight.value = (bodyHeightPixel.value +
    //                 (footPosition.value - footPositionMidTablet) * 1.33) *
    //             heightPerPixelTablet +
    //         11.0
    //     : bodyHeight.value = (bodyHeightPixel.value +
    //                 (footPosition.value - footPositionMidKiosk) * 1.33) *
    //             heightPerPixelKiosk +
    //         11.0;

    final isCorrectPose = _isCorrectPose();

    if (isCorrectPose && !isHoldingPosition && currentState.value == 4) {
      _startHoldingPosition();
    } else if (!isCorrectPose) {
      _stopHoldingPosition();
    }

    posePaint.value = CustomPaint(
      painter: PosePainter(
        poses,
        Size(image.width.toDouble(), image.height.toDouble()),
        isCorrectPose,
      ),
    );

    circularIndicatorPaint.value = CustomPaint(
      painter: CircularIndicatorPainter(holdTime.value),
    );
  }

  bool _isCorrectPose() {
    final correctFootPosition = isTablet.value
        ? footPosition.value >= footPositionMinTablet &&
            footPosition.value <= footPositionMaxTablet
        : footPosition.value >= footPositionMinKiosk &&
            footPosition.value <= footPositionMaxKiosk;

    return correctFootPosition;

    // return correctFootPosition &&
    //     _isHeelsOnGround(leftHeel, leftFoot, rightHeel, rightFoot);
  }

  // bool _isHeelsOnGround(PoseLandmark leftHeel, PoseLandmark leftFoot,
  //     PoseLandmark rightHeel, PoseLandmark rightFoot) {
  //   // 발의 각도를 계산하여 발 뒤꿈치가 들리지 않았는지 확인
  //   double leftHeelAngle = _calculateAngle(
  //     Offset(leftHeel.x, leftHeel.y),
  //     Offset(leftFoot.x, leftFoot.y),
  //     Offset(leftFoot.x + 1, leftFoot.y),
  //   );

  //   double rightHeelAngle = _calculateAngle(
  //     Offset(rightHeel.x, rightHeel.y),
  //     Offset(rightFoot.x, rightFoot.y),
  //     Offset(rightFoot.x + 1, rightFoot.y),
  //   );

  //   isHeelGround.value = leftHeelAngle <= heelAngleThreshold &&
  //       rightHeelAngle <= heelAngleThreshold;

  //   leftHeelAngleDebug.value = leftHeelAngle;
  //   rightHeelAngleDebug.value = rightHeelAngle;

  //   // 각도가 heelAngleThreshold 이하이면 발 뒤꿈치가 지면에 있다고 판단
  //   return leftHeelAngle <= heelAngleThreshold &&
  //       rightHeelAngle <= heelAngleThreshold;
  // }

  void _startHoldingPosition() {
    isHoldingPosition = true;
    heightMeasurements.clear();
    holdTime.value = 0;
    holdSignalState.value = 0;

    // 3초 동안 높이 측정 시작 및 음성 신호 재생
    holdTimer =
        Timer.periodic(const Duration(milliseconds: 100), (timer) async {
      holdTime.value += 0.1;
      heightMeasurements.add(bodyHeight.value);

      if (holdTime.value >= 0.0 && holdSignalState.value == 0) {
        holdSignalState.value = 1;
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/count_down_3.mp3"));
      } else if (holdTime.value >= 1.0 && holdSignalState.value == 1) {
        holdSignalState.value = 2;
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/count_down_2.mp3"));
      } else if (holdTime.value >= 2.0 && holdSignalState.value == 2) {
        holdSignalState.value = 3;
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/count_down_1.mp3"));
      }

      if (holdTime.value >= 3.0) {
        currentState.value = 5;
        _calculateAverageHeight();
        _stopHoldingPosition();
        timer.cancel();
        await bgmPlayer.stop();
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/whistle_done.mp3"));
        signalPeriodical = Timer(const Duration(seconds: 1), () async {
          if (isLocal.value == false && profileId.value > 0) {
            final trackingId = (await service.addTrackingOrganization(
                400, 400, 500))['tracking_id'];
            service.addTrackingDataOrganization(
                trackingId, 240, (bodyHeight.value * 100).toInt());
          }
          Get.offNamed('/mission/measurement/result',
              arguments: MeasurementResultData(
                resultNum: bodyHeight.value,
                grade: 0,
                testType: 5,
              ));
        });
      }
    });
  }

  void _stopHoldingPosition() {
    holdTimer?.cancel();
    isHoldingPosition = false;
    holdTime.value = 0;
  }

  void _calculateAverageHeight() {
    if (heightMeasurements.isNotEmpty) {
      bodyHeight.value = heightMeasurements.reduce((a, b) => a + b) /
          heightMeasurements.length;
    }
  }

  @override
  void onClose() {
    holdTimer?.cancel();
    super.onClose();
  }
}

class MotionSargeantJumpController extends PoseDetectionController {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(),
  );

  static const int footPositionMinKiosk = 820;
  static const int footPositionMaxKiosk = 920;
  static const int footPositionMinTablet = 195;
  static const int footPositionMaxTablet = 215;

  final jumpHeight = 0.0.obs;
  final footPosition = 0.0.obs;

  final isReady = false.obs;
  final isJump = false.obs;

  DateTime startTime = DateTime.now();
  DateTime endTime = DateTime.now();

  final duration = 0.0.obs;

  @override
  void _initCamera() async {
    final List<CameraDescription> cameras = await availableCameras();
    final CameraDescription camera = isTablet.value
        ? cameras.firstWhere(
            (element) => element.lensDirection == CameraLensDirection.front)
        : cameras[0];

    cameraController = isTablet.value
        ? CameraController(
            camera,
            ResolutionPreset.low,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          )
        : CameraController(
            camera,
            ResolutionPreset.max,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.nv21,
          );

    cameraController!.initialize().then((value) {
      isCameraInit.value = true;
      cameraController!.startImageStream(_imageAnalysis);
    });
  }

  void _imageAnalysis(CameraImage cameraImage) async {
    if (_isFrameProcessing.value) return;

    _isFrameProcessing.value = true;
    final inputImage = await _convertCameraImageToInputImage(cameraImage);
    final poses = await _poseDetector.processImage(inputImage);
    _processPoses(poses, cameraImage);
    _isFrameProcessing.value = false;
  }

  @override
  Future<void> playBgm() async {
    await bgmPlayer.play(
        AssetSource('lib/module/measurement/assets/measurement_02_bgm.mp3'));
  }

  void _processPoses(List<Pose> poses, CameraImage image) async {
    if (poses.isEmpty) return;

    final pose = poses.first;
    final leftHeel = pose.landmarks[PoseLandmarkType.leftHeel];
    final rightHeel = pose.landmarks[PoseLandmarkType.rightHeel];
    final leftFootIndex = pose.landmarks[PoseLandmarkType.leftFootIndex];
    final rightFootIndex = pose.landmarks[PoseLandmarkType.rightFootIndex];
    final leftEye = pose.landmarks[PoseLandmarkType.leftEye];
    final rightEye = pose.landmarks[PoseLandmarkType.rightEye];

    if (leftHeel == null ||
        rightHeel == null ||
        leftFootIndex == null ||
        rightFootIndex == null ||
        leftEye == null ||
        rightEye == null) return;

    footPosition.value = (leftFootIndex.y + rightFootIndex.y) / 2.0;

    final isCorrectPose = _isCorrectPose();

    if (isCorrectPose && !isJump.value && currentState.value == 4) {
      isReady.value = true;
    } else if (!isCorrectPose &&
        isReady.value &&
        !isJump.value &&
        currentState.value == 4) {
      isReady.value = false;
      isJump.value = true;
      startTime = DateTime.now();
      await signalPlayer
          .play(AssetSource("lib/module/measurement/assets/whistle_short.mp3"));
    } else if (isCorrectPose && isJump.value && currentState.value == 4) {
      isJump.value = false;
      endTime = DateTime.now();
      duration.value = (endTime.difference(startTime).inMilliseconds) / 1000;
      if (duration.value < 0.3 || duration.value > 1.0) {
        duration.value = 0.0;
      } else {
        currentState.value = 5;
        await bgmPlayer.stop();
        await signalPlayer.play(
            AssetSource("lib/module/measurement/assets/whistle_done.mp3"));

        final grade = await _calculateGrade(duration.value);
        signalPeriodical = Timer(const Duration(seconds: 1), () async {
          if (isLocal.value == false && profileId.value > 0) {
            final trackingId = (await service.addTrackingOrganization(
                400, 400, 300))['tracking_id'];
            service.addTrackingDataOrganization(
                trackingId, 110, stopwatch.elapsed.inSeconds);
            service.addTrackingDataOrganization(
                trackingId, 220, (duration.value * 1000).toInt());
            service.addTrackingDataOrganization(trackingId, 221, grade);
          }
          Get.offNamed('/mission/measurement/result',
              arguments: MeasurementResultData(
                resultNum: duration.value,
                grade: grade,
                testType: 3,
              ));
        });
      }
    }

    posePaint.value = CustomPaint(
      painter: PosePainter(
        poses,
        Size(image.width.toDouble(), image.height.toDouble()),
        isCorrectPose,
      ),
    );
  }

  bool _isCorrectPose() {
    final correctFootPosition = isTablet.value
        ? footPosition.value >= footPositionMinTablet &&
            footPosition.value <= footPositionMaxTablet
        : footPosition.value >= footPositionMinKiosk &&
            footPosition.value <= footPositionMaxKiosk;

    return correctFootPosition;
  }

  Future<int> _calculateGrade(double value) async {
    String jsonString = await rootBundle
        .loadString('lib/module/measurement/assets/measurement_grade.json');
    final jsonResponse = await json.decode(jsonString);

    if (value >= jsonResponse[2][userMetricsType]["1"]) return 1;
    if (value >= jsonResponse[2][userMetricsType]["2"]) return 2;
    if (value >= jsonResponse[2][userMetricsType]["3"]) return 3;
    if (value >= jsonResponse[2][userMetricsType]["4"]) return 4;
    return 5;
  }
}

class MeasurementWeightController extends GetxController {
  static MeasurementWeightController get to => Get.find();

  final isInitialized = false.obs;
  final isVodLoaded = false.obs;
  VideoPlayerController? videoPlayerController;
  ChewieController? chewieController;

  @override
  void onInit() {
    super.onInit();
    init();
  }

  @override
  void onClose() {
    chewieController?.dispose();
    videoPlayerController?.dispose();
    recentWeight.value = 0;
    super.onClose();
  }

  void init() async {
    videoPlayerController = VideoPlayerController.asset(
        'lib/module/measurement/assets/measurement_06.mp4');
    chewieController = ChewieController(
      videoPlayerController: videoPlayerController!,
      aspectRatio: 16 / 9,
      autoPlay: true,
      autoInitialize: true,
      showControlsOnInitialize: false,
      showControls: false,
      looping: true,
    );
    isInitialized.value = true;
    videoPlayerController?.addListener(() {
      if (isVodLoaded.value == false) {
        isVodLoaded.value = true;
      }
    });
  }

  void saveWeight() async {
    if (recentWeight.value != 0 &&
        isLocal.value == false &&
        profileId.value > 0) {
      final trackingId =
          (await service.addTrackingOrganization(200, 400, 600))['tracking_id'];
      service.addTrackingDataOrganization(trackingId, 100, recentWeight.value);

      recentWeight.value = 0;

      Get.back();
      getSnackBar("알림", "체중이 저장되었습니다");
    } else {
      getSnackBar("알림", "로그인이 필요합니다");
    }
  }
}
