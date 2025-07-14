import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(AgriScanAIApp());
}

class AgriScanAIApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AgriScanAI',
      theme: ThemeData(primarySwatch: Colors.green),
      home: HomeScreen(),
    );
  }
}
