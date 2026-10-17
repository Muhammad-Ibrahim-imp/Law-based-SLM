import 'package:flutter_test/flutter_test.dart';
import 'package:legal_chat/models.dart';

void main() {
  test(
    'attachments round trip and legacy messages default to no attachments',
    () {
      final message = ChatMessage(
        id: 'm',
        role: MessageRole.user,
        text: 'Review this file',
        createdAt: DateTime.utc(2026),
        attachments: const [
          ChatAttachment(
            localPath: '/private/brief.pdf',
            name: 'brief.pdf',
            mediaType: 'application/pdf',
          ),
        ],
      );
      final copy = ChatMessage.fromJson(message.toJson());
      expect(copy.attachments.single.localPath, '/private/brief.pdf');
      expect(copy.attachments.single.name, 'brief.pdf');
      expect(copy.attachments.single.mediaType, 'application/pdf');

      final legacy = ChatMessage.fromJson({
        'id': 'old',
        'role': 'user',
        'text': 'Older saved prompt',
        'createdAt': DateTime.utc(2025).toIso8601String(),
      });
      expect(legacy.attachments, isEmpty);
    },
  );

  test(
    'workspace round trips projects, chat metadata, and markdown prompts',
    () {
      final original = Workspace(
        projects: [ChatProject(id: 'p', name: 'Cases')],
        chats: [
          Chat(
            id: 'c',
            title: 'Question',
            updatedAt: DateTime.utc(2026),
            projectId: 'p',
            pinned: true,
            messages: [
              ChatMessage(
                id: 'm',
                role: MessageRole.user,
                text: '**Keep this prompt**',
                createdAt: DateTime.utc(2026),
              ),
            ],
          ),
        ],
      );
      final copy = Workspace.decode(original.encode());
      expect(copy.projects.single.name, 'Cases');
      expect(copy.chats.single.pinned, isTrue);
      expect(copy.chats.single.messages.single.text, '**Keep this prompt**');
    },
  );
}
