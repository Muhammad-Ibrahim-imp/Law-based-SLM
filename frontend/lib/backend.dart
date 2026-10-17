import 'dart:async';
import 'models.dart';

abstract interface class ChatBackend {
  bool get isAvailable;
  bool get supportsAttachments;
  String get unavailableReason;
  Stream<String> generate(List<ChatMessage> conversation);
  void cancel();
}

class UnconfiguredBackend implements ChatBackend {
  @override
  bool get isAvailable => false;
  @override
  bool get supportsAttachments => false;
  @override
  String get unavailableReason =>
      'No AI model is installed. Your prompts are saved locally; replies are unavailable until a model is added.';
  @override
  Stream<String> generate(List<ChatMessage> conversation) =>
      const Stream.empty();
  @override
  void cancel() {}
}
