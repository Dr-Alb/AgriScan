import 'package:flutter/material.dart';
import '../widgets/image_input.dart';

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('AgriScanAI')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Scan a plant leaf to detect diseases', style: TextStyle(fontSize: 18), textAlign: TextAlign.center),
              SizedBox(height: 20),
              ImageInput(),
            ],
          ),
        ),
      ),
    );
  }
}
