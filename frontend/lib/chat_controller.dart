import 'dart:async';
import 'package:flutter/foundation.dart';
import 'models.dart';
import 'backend.dart';
import 'secure_store.dart';

class ChatController extends ChangeNotifier {
  ChatController(this.store, this.backend);
  final WorkspaceStore store;
  final ChatBackend backend;
  Workspace workspace = Workspace();
  String? activeChatId;
  String? activeProjectId;
  String? error;
  bool loading = false;
  bool initialized = false;
  bool locked = false;
  bool _disposed = false;
  bool _busy = false;
  StreamSubscription<String>? _generation;
  Completer<void>? _generationDone;
  Completer<void>? _generationFinished;
  Future<void> _saveQueue = Future<void>.value();
  bool get isGenerating => _generation != null;
  bool get isBusy => _busy;
  bool get canWrite => initialized && !loading && !locked;
  Chat? get activeChat {
    for (final chat in workspace.chats) {
      if (chat.id == activeChatId) return chat;
    }
    return null;
  }

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  Future<void> initialize() async {
    if (initialized || loading) return;
    loading = true;
    _notify();
    try {
      workspace = await store.load();
    } catch (e) {
      error =
          'Could not open encrypted chats. Data is locked and preserved: $e';
      locked = true;
    } finally {
      loading = false;
      initialized = true;
      _notify();
    }
  }

  void _ensureWritable() {
    if (!canWrite) throw StateError(error ?? 'Workspace is not ready');
  }

  Future<bool> _save() async {
    _ensureWritable();
    final snapshot = Workspace.decode(workspace.encode());
    final save = _saveQueue.then((_) => store.save(snapshot));
    _saveQueue = save.catchError((Object _) {});
    try {
      await save;
      if (error?.startsWith('Could not save changes.') == true) error = null;
      _notify();
      return true;
    } catch (e) {
      error =
          'Could not save changes. These edits are only in memory for this session; saved data was left intact: $e';
      _notify();
      return false;
    }
  }

  Future<Chat> createChat() async {
    _ensureWritable();
    final c = Chat(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      title: 'New chat',
      updatedAt: DateTime.now(),
      projectId: activeProjectId,
    );
    workspace.chats.insert(0, c);
    activeChatId = c.id;
    await _save();
    _notify();
    return c;
  }

  void selectChat(String id) {
    activeChatId = id;
    _notify();
  }

  void selectProject(String id) {
    activeProjectId = id;
    activeChatId = null;
    _notify();
  }

  void showAllChats() {
    activeProjectId = null;
    activeChatId = null;
    _notify();
  }

  Future<void> renameChat(String id, String title) async {
    _ensureWritable();
    final c = workspace.chats.firstWhere((x) => x.id == id);
    c.title = title.trim().isEmpty ? c.title : title.trim();
    await _save();
  }

  Future<void> deleteChat(String id) async {
    _ensureWritable();
    workspace.chats.removeWhere((c) => c.id == id);
    if (activeChatId == id) activeChatId = null;
    await _save();
  }

  Future<void> togglePin(String id) async {
    _ensureWritable();
    final c = workspace.chats.firstWhere((x) => x.id == id);
    c.pinned = !c.pinned;
    await _save();
  }

  Future<void> createProject(String name) async {
    _ensureWritable();
    workspace.projects.add(
      ChatProject(
        id: DateTime.now().microsecondsSinceEpoch.toString(),
        name: name.trim(),
      ),
    );
    await _save();
  }

  Future<void> renameProject(String id, String name) async {
    _ensureWritable();
    if (name.trim().isEmpty) return;
    workspace.projects.firstWhere((p) => p.id == id).name = name.trim();
    await _save();
  }

  Future<void> deleteProject(String id) async {
    _ensureWritable();
    workspace.projects.removeWhere((p) => p.id == id);
    for (final c in workspace.chats) {
      if (c.projectId == id) c.projectId = null;
    }
    if (activeProjectId == id) activeProjectId = null;
    await _save();
  }

  Future<void> moveChat(String chatId, String? projectId) async {
    _ensureWritable();
    workspace.chats.firstWhere((c) => c.id == chatId).projectId = projectId;
    await _save();
  }

  Future<void> setThemeMode(String value) async {
    _ensureWritable();
    workspace.themeMode = value;
    await _save();
  }

  Future<void> send(String prompt) async {
    _ensureWritable();
    if (_busy) return;
    final text = prompt.trim();
    if (text.isEmpty) return;
    _busy = true;
    error = null;
    _notify();
    try {
      final chat = activeChat ?? await createChat();
      chat.messages.add(
        ChatMessage(
          id: DateTime.now().microsecondsSinceEpoch.toString(),
          role: MessageRole.user,
          text: text,
          createdAt: DateTime.now(),
        ),
      );
      if (chat.title == 'New chat') {
        chat.title = text.length > 36 ? '${text.substring(0, 36)}…' : text;
      }
      chat.updatedAt = DateTime.now();
      final saved = await _save();
      _notify();
      if (saved && backend.isAvailable) await _generate(chat);
    } finally {
      _busy = false;
      _notify();
    }
  }

  Future<void> _generate(Chat chat) async {
    if (error?.startsWith('Generation ') == true) error = null;
    final reply = ChatMessage(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      role: MessageRole.assistant,
      text: '',
      createdAt: DateTime.now(),
    );
    chat.messages.add(reply);
    final done = _generationDone = Completer<void>();
    final finished = _generationFinished = Completer<void>();
    _notify();
    try {
      _generation = backend
          .generate(
            List.unmodifiable(chat.messages.take(chat.messages.length - 1)),
          )
          .listen(
            (part) {
              reply.text += part;
              _notify();
            },
            onError: (Object e) {
              error = 'Generation failed: $e';
              if (!done.isCompleted) done.complete();
            },
            onDone: () {
              if (!done.isCompleted) done.complete();
            },
            cancelOnError: true,
          );
      await done.future;
    } catch (e) {
      error = 'Generation failed: $e';
      if (!done.isCompleted) done.complete();
    } finally {
      _generation = null;
      _generationDone = null;
      await _save();
      _notify();
      _generationFinished = null;
      if (!finished.isCompleted) finished.complete();
    }
  }

  Future<void> stop() async {
    final sub = _generation;
    final finished = _generationFinished;
    if (sub == null || finished == null) return;
    try {
      backend.cancel();
    } catch (e) {
      error = 'Generation cancellation failed: $e';
    }
    await sub.cancel();
    final done = _generationDone;
    if (done != null && !done.isCompleted) done.complete();
    await finished.future;
  }

  Future<void> regenerate() async {
    final c = activeChat;
    if (!canWrite ||
        _busy ||
        c == null ||
        c.messages.isEmpty ||
        !backend.isAvailable) {
      return;
    }
    _busy = true;
    _notify();
    try {
      if (c.messages.last.role == MessageRole.assistant) {
        c.messages.removeLast();
      }
      if (c.messages.isNotEmpty && c.messages.last.role == MessageRole.user) {
        await _generate(c);
      }
    } finally {
      _busy = false;
      _notify();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    unawaited(stop());
    super.dispose();
  }
}
