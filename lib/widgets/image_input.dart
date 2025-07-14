import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/tflite_service.dart';

class ImageInput extends StatefulWidget {
  @override
  _ImageInputState createState() => _ImageInputState();
}

class _ImageInputState extends State<ImageInput> {
  File? _image;
  String _result = '';
  final tfliteService = TFLiteService();

  @override
  void initState() {
    super.initState();
    tfliteService.loadModel();
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.camera);
    if (picked == null) return;

    setState(() => _image = File(picked.path));
    final prediction = await tfliteService.classifyImage(_image!);
    setState(() => _result = prediction);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (_image != null) Image.file(_image!, height: 200),
        ElevatedButton.icon(
          icon: Icon(Icons.camera),
          label: Text('Capture Leaf'),
          onPressed: _pickImage,
        ),
        if (_result.isNotEmpty) Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(_result, style: TextStyle(fontSize: 18)),
        )
      ],
    );
  }
}
