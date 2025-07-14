import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class TFLiteService {
  late Interpreter _interpreter;
  late List<String> _labels;
  final int inputSize = 224;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('plant_disease_model.tflite');
    final labelsData = await rootBundle.loadString('assets/label_map.txt');
    _labels = labelsData.split('\n');
  }

  Future<String> classifyImage(File imageFile) async {
    final image = TensorImage.fromFile(imageFile);
    final processor = ImageProcessorBuilder()
        .add(ResizeOp(inputSize, inputSize, ResizeMethod.BILINEAR))
        .build();
    final input = processor.process(image).buffer;
    final output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
    _interpreter.run(input, output);
    final result = output[0] as List;
    final maxIdx = result.indexWhere((e) => e == result.reduce((a, b) => a > b ? a : b));
    return '\${_labels[maxIdx]} (confidence: \${(result[maxIdx] * 100).toStringAsFixed(2)}%)';
  }
}
