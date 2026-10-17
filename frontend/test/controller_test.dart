import 'package:flutter_test/flutter_test.dart';
import 'dart:async';
import 'package:legal_chat/backend.dart';
import 'package:legal_chat/chat_controller.dart';
import 'package:legal_chat/models.dart';
import 'package:legal_chat/secure_store.dart';

class MemoryStore implements WorkspaceStore {
  Workspace data = Workspace();
  bool failLoad = false;
  bool failSave = false;
  @override
  Future<Workspace> load() async {
    if (failLoad) throw const FormatException('corrupt');
    return data;
  }

  @override
  Future<void> save(Workspace value) async {
    if (failSave) throw StateError('disk full');
    data = Workspace.decode(value.encode());
  }
}

class DelayedLoadStore implements WorkspaceStore {
  DelayedLoadStore(this.data);
  final Workspace data;
  final entered = Completer<void>();
  final release = Completer<void>();
  int saves = 0;

  @override
  Future<Workspace> load() async {
    entered.complete();
    await release.future;
    return data;
  }

  @override
  Future<void> save(Workspace value) async {
    saves++;
  }
}

class DelayedSaveStore implements WorkspaceStore {
  Workspace data = Workspace();
  final entered = Completer<void>();
  final release = Completer<void>();
  int saves = 0;

  @override
  Future<Workspace> load() async => data;

  @override
  Future<void> save(Workspace value) async {
    saves++;
    if (saves == 1) {
      entered.complete();
      await release.future;
    }
    data = Workspace.decode(value.encode());
  }
}

class FakeBackend implements ChatBackend {
  @override
  bool get supportsAttachments => false;
  @override
  bool get isAvailable => false;
  @override
  String get unavailableReason => 'No model';
  @override
  Stream<String> generate(List<ChatMessage> conversation) =>
      const Stream.empty();
  @override
  void cancel() {}
}

class StreamingBackend implements ChatBackend {
  @override
  bool get supportsAttachments => false;
  StreamController<String>? output;
  final requests = <List<ChatMessage>>[];
  @override
  bool get isAvailable => true;
  @override
  String get unavailableReason => '';
  @override
  Stream<String> generate(List<ChatMessage> conversation) {
    requests.add(List.of(conversation));
    output = StreamController<String>();
    return output!.stream;
  }

  @override
  void cancel() {}
}

void main() {
  test('writes are blocked until delayed initialization completes', () async {
    final storedChat = Chat(
      id: 'saved',
      title: 'Saved history',
      updatedAt: DateTime.utc(2026),
    );
    final store = DelayedLoadStore(Workspace(chats: [storedChat]));
    final controller = ChatController(store, FakeBackend());
    final loading = controller.initialize();
    await store.entered.future;
    expect(controller.canWrite, isFalse);
    await expectLater(controller.createChat(), throwsA(isA<StateError>()));
    await expectLater(
      controller.send('Do not overwrite'),
      throwsA(isA<StateError>()),
    );
    expect(store.saves, 0);
    store.release.complete();
    await loading;
    expect(controller.workspace.chats.single.title, 'Saved history');
    expect(store.saves, 0);
  });

  test('concurrent send while initial save is pending is ignored', () async {
    final store = DelayedSaveStore();
    final controller = ChatController(store, FakeBackend());
    await controller.initialize();
    final first = controller.send('First prompt');
    await store.entered.future;
    await controller.send('Duplicate prompt');
    store.release.complete();
    await first;
    expect(store.data.chats.single.messages.map((m) => m.text), [
      'First prompt',
    ]);
    expect(store.saves, 2); // New chat creation and the prompt are both saved.
  });

  test(
    'local controller saves user prompts, project moves, pins, rename and deletes',
    () async {
      final store = MemoryStore();
      final controller = ChatController(store, FakeBackend());
      await controller.initialize();
      await controller.createProject('Work');
      final project = controller.workspace.projects.single;
      final chat = await controller.createChat();
      await controller.send('Question');
      await controller.moveChat(chat.id, project.id);
      await controller.togglePin(chat.id);
      await controller.renameChat(chat.id, 'Matter');
      expect(store.data.chats.single.messages.single.text, 'Question');
      expect(store.data.chats.single.projectId, project.id);
      expect(store.data.chats.single.pinned, isTrue);
      expect(store.data.chats.single.title, 'Matter');
      await controller.deleteProject(project.id);
      expect(store.data.chats.single.projectId, isNull);
      await controller.deleteChat(chat.id);
      expect(store.data.chats, isEmpty);
    },
  );
  test(
    'failed load locks the workspace and leaves stored bytes alone',
    () async {
      final store = MemoryStore()..failLoad = true;
      final controller = ChatController(store, FakeBackend());
      await controller.initialize();
      expect(controller.locked, isTrue);
      expect(controller.workspace.chats, isEmpty);
      expect(controller.error, contains('preserved'));
      expect(controller.createChat(), throwsA(isA<StateError>()));
    },
  );
  test(
    'stream receives role-aware history, stop persists partial output, regenerate does not duplicate prompt',
    () async {
      final store = MemoryStore();
      final backend = StreamingBackend();
      final controller = ChatController(store, backend);
      await controller.initialize();
      final pending = controller.send('Question');
      await Future<void>.delayed(Duration.zero);
      backend.output!.add('partial');
      await Future<void>.delayed(Duration.zero);
      expect(controller.activeChat!.messages.last.text, 'partial');
      await controller.stop();
      await pending;
      expect(store.data.chats.single.messages.map((m) => m.text), [
        'Question',
        'partial',
      ]);
      final regeneration = controller.regenerate();
      await Future<void>.delayed(Duration.zero);
      expect(backend.requests.last.map((m) => m.role), [MessageRole.user]);
      expect(backend.requests.last.map((m) => m.text), ['Question']);
      backend.output!.add('complete');
      await backend.output!.close();
      await regeneration;
      expect(store.data.chats.single.messages.map((m) => m.text), [
        'Question',
        'complete',
      ]);
    },
  );
  test(
    'save failures keep the prior persisted copy and surface an error',
    () async {
      final store = MemoryStore();
      final controller = ChatController(store, FakeBackend());
      await controller.initialize();
      final chat = await controller.createChat();
      store.failSave = true;
      await controller.renameChat(chat.id, 'Unsaved rename');
      expect(store.data.chats.single.title, 'New chat');
      expect(controller.error, contains('only in memory'));
    },
  );
  test(
    'generation errors are visible and retain the saved user prompt',
    () async {
      final store = MemoryStore();
      final backend = StreamingBackend();
      final controller = ChatController(store, backend);
      await controller.initialize();
      final pending = controller.send('Question');
      await Future<void>.delayed(Duration.zero);
      backend.output!.addError(StateError('model failed'));
      await pending;
      expect(controller.error, contains('Generation failed'));
      expect(store.data.chats.single.messages.first.text, 'Question');
    },
  );
  test('regeneration includes preceding assistant context', () async {
    final store = MemoryStore()
      ..data = Workspace(
        chats: [
          Chat(
            id: 'c',
            title: 'Matter',
            updatedAt: DateTime.utc(2026),
            messages: [
              ChatMessage(
                id: '1',
                role: MessageRole.user,
                text: 'First',
                createdAt: DateTime.utc(2026),
              ),
              ChatMessage(
                id: '2',
                role: MessageRole.assistant,
                text: 'Context',
                createdAt: DateTime.utc(2026),
              ),
              ChatMessage(
                id: '3',
                role: MessageRole.user,
                text: 'Follow up',
                createdAt: DateTime.utc(2026),
              ),
              ChatMessage(
                id: '4',
                role: MessageRole.assistant,
                text: 'Old answer',
                createdAt: DateTime.utc(2026),
              ),
            ],
          ),
        ],
      );
    final backend = StreamingBackend();
    final controller = ChatController(store, backend);
    await controller.initialize();
    controller.selectChat('c');
    final regeneration = controller.regenerate();
    await Future<void>.delayed(Duration.zero);
    expect(backend.requests.single.map((m) => m.role), [
      MessageRole.user,
      MessageRole.assistant,
      MessageRole.user,
    ]);
    backend.output!.add('New answer');
    await backend.output!.close();
    await regeneration;
  });
}
