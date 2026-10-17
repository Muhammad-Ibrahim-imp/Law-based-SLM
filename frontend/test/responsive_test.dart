import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:legal_chat/backend.dart';
import 'package:legal_chat/chat_controller.dart';
import 'package:legal_chat/main.dart';
import 'package:legal_chat/models.dart';
import 'package:legal_chat/secure_store.dart';

class MemoryStore implements WorkspaceStore {
  MemoryStore([Workspace? workspace]) : workspace = workspace ?? Workspace();
  Workspace workspace;
  @override
  Future<Workspace> load() async => workspace;
  @override
  Future<void> save(Workspace value) async => workspace = value;
}

class NoModel implements ChatBackend {
  @override
  bool get supportsAttachments => false;
  @override
  bool get isAvailable => false;
  @override
  String get unavailableReason => 'No model installed';
  @override
  Stream<String> generate(List<ChatMessage> conversation) =>
      const Stream.empty();
  @override
  void cancel() {}
}

void main() {
  testWidgets('small layout uses a drawer and shows the no-model state', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(390, 844);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      MaterialApp(
        home: HomePage(
          controller: ChatController(MemoryStore(), NoModel()),
          mode: ThemeMode.dark,
          onMode: (_) {},
        ),
      ),
    );
    expect(find.byIcon(Icons.menu), findsOneWidget);
    expect(find.text('How can I help you today?'), findsOneWidget);
    await tester.tap(find.byIcon(Icons.menu));
    await tester.pumpAndSettle();
    expect(find.text('Legal Chat'), findsOneWidget);
  });
  testWidgets('wide layout keeps the sidebar visible without overflow', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1280, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      MaterialApp(
        home: HomePage(
          controller: ChatController(MemoryStore(), NoModel()),
          mode: ThemeMode.dark,
          onMode: (_) {},
        ),
      ),
    );
    expect(find.text('Legal Chat'), findsOneWidget);
    expect(find.byIcon(Icons.menu), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('short mobile drawer scrolls long project and chat lists', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(844, 390);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final workspace = Workspace(
      projects: [
        for (var i = 0; i < 20; i++) ChatProject(id: 'p$i', name: 'Project $i'),
      ],
      chats: [
        for (var i = 0; i < 20; i++)
          Chat(
            id: 'c$i',
            title: 'Chat $i',
            updatedAt: DateTime(2026, 1, i + 1),
          ),
      ],
    );
    final controller = ChatController(MemoryStore(workspace), NoModel());
    await controller.initialize();
    await tester.pumpWidget(
      MaterialApp(
        home: HomePage(
          controller: controller,
          mode: ThemeMode.dark,
          onMode: (_) {},
        ),
      ),
    );
    await tester.tap(find.byIcon(Icons.menu));
    await tester.pumpAndSettle();
    expect(find.text('Project 0'), findsOneWidget);
    final sidebarScroll = find.descendant(
      of: find.byType(Drawer),
      matching: find.byType(Scrollable),
    );
    final scrollState = tester.state<ScrollableState>(sidebarScroll.first);
    expect(scrollState.position.maxScrollExtent, greaterThan(0));
    scrollState.position.jumpTo(scrollState.position.maxScrollExtent);
    await tester.pumpAndSettle();
    expect(find.text('Chat 0'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
