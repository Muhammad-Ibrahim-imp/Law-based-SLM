import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'backend.dart';
import 'chat_controller.dart';
import 'models.dart';
import 'secure_store.dart';

void main() => runApp(const LegalChatApp());

class LegalChatApp extends StatefulWidget {
  const LegalChatApp({super.key});
  @override
  State<LegalChatApp> createState() => _LegalChatAppState();
}

class _LegalChatAppState extends State<LegalChatApp> {
  late final ChatController controller = ChatController(
    EncryptedWorkspaceStore(),
    UnconfiguredBackend(),
  );
  ThemeMode mode = ThemeMode.dark;
  @override
  void initState() {
    super.initState();
    controller.initialize().then((_) {
      if (!mounted) return;
      final stored = ThemeMode.values.where(
        (m) => m.name == controller.workspace.themeMode,
      );
      setState(() => mode = stored.isEmpty ? ThemeMode.dark : stored.first);
    });
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  ThemeData theme(Brightness b) {
    final dark = b == Brightness.dark;
    final scheme = ColorScheme.fromSeed(
      seedColor: const Color(0xff8eb6ff),
      brightness: b,
      surface: dark ? const Color(0xff101010) : const Color(0xfffaf9f7),
    );
    return ThemeData(
      colorScheme: scheme,
      brightness: b,
      scaffoldBackgroundColor: dark
          ? const Color(0xff101010)
          : const Color(0xfffaf9f7),
      fontFamily: 'Fougie',
      useMaterial3: true,
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      cardTheme: CardThemeData(
        color: dark ? const Color(0xff202020) : Colors.white,
        elevation: 0,
      ),
    );
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Legal Chat',
    theme: theme(Brightness.light),
    darkTheme: theme(Brightness.dark),
    themeMode: mode,
    home: HomePage(
      controller: controller,
      mode: mode,
      onMode: (m) {
        setState(() => mode = m);
        if (controller.canWrite) {
          unawaited(controller.setThemeMode(m.name));
        }
      },
    ),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({
    super.key,
    required this.controller,
    required this.mode,
    required this.onMode,
  });
  final ChatController controller;
  final ThemeMode mode;
  final ValueChanged<ThemeMode> onMode;
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final input = TextEditingController();
  ChatController get c => widget.controller;
  bool get mobile => MediaQuery.sizeOf(context).width < 900;
  @override
  void initState() {
    super.initState();
    c.addListener(_changed);
  }

  @override
  void dispose() {
    c.removeListener(_changed);
    input.dispose();
    super.dispose();
  }

  void _changed() {
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final sidebar = _sidebar(context);
    return Scaffold(
      drawer: mobile ? Drawer(child: sidebar) : null,
      body: SafeArea(
        child: Row(
          children: [
            if (!mobile) SizedBox(width: 280, child: sidebar),
            Expanded(
              child: Column(
                children: [
                  _topbar(),
                  Expanded(child: _conversation()),
                  _composer(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _topbar() => SizedBox(
    height: 64,
    child: Row(
      children: [
        if (mobile)
          Builder(
            builder: (ctx) => IconButton(
              onPressed: () => Scaffold.of(ctx).openDrawer(),
              icon: const Icon(Icons.menu),
            ),
          ),
        const SizedBox(width: 12),
        PopupMenuButton<String>(
          tooltip: 'Model',
          onSelected: (_) {},
          itemBuilder: (_) => const [
            PopupMenuItem(enabled: false, child: Text('No model installed')),
            PopupMenuItem(enabled: false, child: Text('Family · unavailable')),
            PopupMenuItem(
              enabled: false,
              child: Text('Criminal · unavailable'),
            ),
            PopupMenuItem(
              enabled: false,
              child: Text('Jev / Laya · unavailable'),
            ),
          ],
          child: Row(
            children: [
              Text(
                'General Chat',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const Icon(Icons.keyboard_arrow_down),
            ],
          ),
        ),
        const Spacer(),
        IconButton(
          onPressed: () => _showShare(),
          icon: const Icon(Icons.ios_share_outlined),
          tooltip: 'Share',
        ),
        PopupMenuButton<String>(
          onSelected: (v) {
            if (v == 'settings') {
              _themeDialog();
            } else {
              _info(v);
            }
          },
          itemBuilder: (_) => const [
            PopupMenuItem(value: 'settings', child: Text('Settings')),
            PopupMenuItem(value: 'help', child: Text('Help')),
            PopupMenuItem(value: 'learn', child: Text('Learn more')),
          ],
          tooltip: 'More',
          child: const Padding(
            padding: EdgeInsets.all(12),
            child: Icon(Icons.more_horiz),
          ),
        ),
      ],
    ),
  );
  Widget _sidebar(BuildContext context) => Material(
    color: Theme.of(context).brightness == Brightness.dark
        ? const Color(0xff202020)
        : const Color(0xfff0efed),
    child: Padding(
      padding: const EdgeInsets.all(14),
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Icon(Icons.balance, size: 21),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(
                    'Legal Chat',
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontFamily: 'Styrene',
                      fontSize: 20,
                    ),
                  ),
                ),
                if (mobile)
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close),
                  ),
              ],
            ),
            const SizedBox(height: 25),
            FilledButton.tonalIcon(
              onPressed: !c.canWrite
                  ? null
                  : () async {
                      await c.createChat();
                      if (!context.mounted) return;
                      if (mobile) Navigator.pop(context);
                    },
              icon: const Icon(Icons.add),
              label: const Text('New chat'),
              style: FilledButton.styleFrom(
                alignment: Alignment.centerLeft,
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 13,
                ),
              ),
            ),
            Expanded(
              child: ListView(
                children: [
                  Row(
                    children: [
                      const Text(
                        'Projects',
                        style: TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const Spacer(),
                      IconButton(
                        onPressed: !c.canWrite ? null : _newProject,
                        icon: const Icon(Icons.add, size: 19),
                        tooltip: 'New project',
                      ),
                    ],
                  ),
                  ...c.workspace.projects.map(
                    (p) => ListTile(
                      dense: true,
                      selected: c.activeProjectId == p.id,
                      leading: const Icon(Icons.folder_outlined, size: 19),
                      title: Text(p.name, overflow: TextOverflow.ellipsis),
                      onTap: () {
                        c.selectProject(p.id);
                        if (mobile) Navigator.pop(context);
                      },
                      trailing: IconButton(
                        icon: const Icon(Icons.more_horiz, size: 19),
                        onPressed: !c.canWrite ? null : () => _projectMenu(p),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          c.activeProjectId == null
                              ? 'Recent'
                              : 'Project chats',
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ),
                      if (c.activeProjectId != null)
                        IconButton(
                          onPressed: c.showAllChats,
                          icon: const Icon(Icons.history, size: 18),
                          tooltip: 'All chats',
                        ),
                    ],
                  ),
                  ..._visibleChats
                      .where((x) => x.pinned)
                      .map((x) => _chatTile(x)),
                  ..._visibleChats
                      .where((x) => !x.pinned)
                      .map((x) => _chatTile(x)),
                ],
              ),
            ),
            const Divider(),
            ListTile(
              leading: const CircleAvatar(child: Icon(Icons.person_outline)),
              title: const Text(
                'Local workspace',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              subtitle: const Text(
                'Account unavailable',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(fontSize: 12),
              ),
              trailing: PopupMenuButton<String>(
                onSelected: (v) {
                  if (v == 'theme' || v == 'settings') {
                    _themeDialog();
                  } else {
                    _info(v);
                  }
                },
                itemBuilder: (_) => const [
                  PopupMenuItem(value: 'settings', child: Text('Settings')),
                  PopupMenuItem(value: 'theme', child: Text('Appearance')),
                  PopupMenuItem(
                    value: 'account',
                    child: Text('Account · unavailable'),
                  ),
                  PopupMenuItem(
                    enabled: false,
                    child: Text('Log out · unavailable'),
                  ),
                  PopupMenuItem(value: 'help', child: Text('Help')),
                  PopupMenuItem(value: 'learn', child: Text('Learn more')),
                ],
                child: const Icon(Icons.more_horiz),
              ),
            ),
          ],
        ),
      ),
    ),
  );
  List<Chat> get _visibleChats =>
      c.workspace.chats
          .where(
            (x) =>
                c.activeProjectId == null || x.projectId == c.activeProjectId,
          )
          .toList()
        ..sort((a, b) => b.updatedAt.compareTo(a.updatedAt));
  Widget _chatTile(Chat chat) => ListTile(
    dense: true,
    selected: c.activeChatId == chat.id,
    leading: Icon(
      chat.pinned ? Icons.push_pin_outlined : Icons.chat_bubble_outline,
      size: 17,
    ),
    title: Text(chat.title, maxLines: 1, overflow: TextOverflow.ellipsis),
    onTap: () {
      c.selectChat(chat.id);
      if (mobile) Navigator.pop(context);
    },
    trailing: PopupMenuButton<String>(
      enabled: c.canWrite,
      onSelected: (v) => _chatAction(chat, v),
      itemBuilder: (_) => [
        const PopupMenuItem(value: 'rename', child: Text('Rename')),
        const PopupMenuItem(value: 'pin', child: Text('Pin / unpin')),
        const PopupMenuItem(value: 'move', child: Text('Move to project')),
        const PopupMenuItem(value: 'delete', child: Text('Delete')),
      ],
      child: const Icon(Icons.more_horiz, size: 17),
    ),
  );
  Widget _conversation() {
    final chat = c.activeChat;
    if (chat == null || chat.messages.isEmpty) {
      final title = c.activeProjectId == null
          ? 'How can I help you today?'
          : c.workspace.projects
                .firstWhere((p) => p.id == c.activeProjectId)
                .name;
      return Center(
        child: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: const Color(0xff8eb6ff).withValues(alpha: .12),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xff8eb6ff).withValues(alpha: .12),
                        blurRadius: 48,
                        spreadRadius: 18,
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.balance,
                    size: 30,
                    color: Color(0xffa8c5ff),
                  ),
                ),
                const SizedBox(height: 28),
                Text(
                  title,
                  style: Theme.of(
                    context,
                  ).textTheme.headlineMedium?.copyWith(fontFamily: 'Styrene'),
                ),
                const SizedBox(height: 12),
                Text(
                  'Your private legal workspace',
                  style: TextStyle(color: Theme.of(context).hintColor),
                ),
                const SizedBox(height: 30),
                if (c.error != null)
                  Text(
                    c.error!,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.orangeAccent),
                  ),
                const SizedBox(height: 12),
                _notice(),
              ],
            ),
          ),
        ),
      );
    }
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 28),
      children: [
        ...chat.messages.map(_message),
        if (c.error != null)
          Text(c.error!, style: const TextStyle(color: Colors.orangeAccent)),
      ],
    );
  }

  Widget _notice() => Container(
    constraints: const BoxConstraints(maxWidth: 500),
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: Theme.of(
        context,
      ).colorScheme.surfaceContainerHighest.withValues(alpha: .55),
      borderRadius: BorderRadius.circular(14),
    ),
    child: Text(
      c.backend.unavailableReason,
      textAlign: TextAlign.center,
      style: TextStyle(color: Theme.of(context).hintColor, fontSize: 13),
    ),
  );
  Widget _message(ChatMessage m) => Align(
    alignment: m.role == MessageRole.user
        ? Alignment.centerRight
        : Alignment.centerLeft,
    child: Container(
      constraints: const BoxConstraints(maxWidth: 760),
      margin: const EdgeInsets.only(bottom: 22),
      padding: EdgeInsets.all(m.role == MessageRole.user ? 14 : 0),
      decoration: BoxDecoration(
        color: m.role == MessageRole.user
            ? Theme.of(context).colorScheme.surfaceContainerHighest
            : Colors.transparent,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (m.role == MessageRole.assistant)
            MarkdownBody(
              data: m.text,
              selectable: true,
              styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context))
                  .copyWith(
                    p: const TextStyle(fontFamily: 'Tiempos', fontSize: 16),
                  ),
            ),
          if (m.role == MessageRole.user) SelectableText(m.text),
          if (m.role == MessageRole.assistant)
            Row(
              children: [
                IconButton(
                  onPressed: () =>
                      Clipboard.setData(ClipboardData(text: m.text)),
                  icon: const Icon(Icons.copy, size: 17),
                  tooltip: 'Copy response',
                ),
                IconButton(
                  onPressed: c.isGenerating ? null : c.regenerate,
                  icon: const Icon(Icons.refresh, size: 17),
                  tooltip: 'Regenerate',
                ),
              ],
            ),
        ],
      ),
    ),
  );
  Widget _composer() => Padding(
    padding: EdgeInsets.fromLTRB(mobile ? 14 : 30, 12, mobile ? 14 : 30, 18),
    child: Align(
      alignment: Alignment.center,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 850),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (c.activeChat != null && c.activeChat!.messages.isNotEmpty)
              _notice(),
            const SizedBox(height: 10),
            Container(
              decoration: BoxDecoration(
                color: Theme.of(
                  context,
                ).colorScheme.surfaceContainerHighest.withValues(alpha: .62),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(
                  color: Theme.of(context).dividerColor.withValues(alpha: .35),
                ),
              ),
              padding: const EdgeInsets.fromLTRB(18, 9, 10, 8),
              child: Column(
                children: [
                  TextField(
                    controller: input,
                    enabled: c.canWrite && !c.isBusy,
                    style: const TextStyle(fontFamily: 'Styrene'),
                    minLines: 1,
                    maxLines: 6,
                    decoration: const InputDecoration(
                      border: InputBorder.none,
                      hintText: 'Message Legal Chat',
                      hintStyle: TextStyle(fontFamily: 'Styrene'),
                    ),
                    onChanged: (_) => setState(() {}),
                    onSubmitted: (_) => _send(),
                  ),
                  Row(
                    children: [
                      IconButton(
                        onPressed: null,
                        tooltip: 'Attachments unavailable',
                        icon: const Icon(Icons.add),
                      ),
                      const Spacer(),
                      FilledButton(
                        onPressed: c.isGenerating
                            ? c.stop
                            : !c.canWrite ||
                                  c.isBusy ||
                                  input.text.trim().isEmpty
                            ? null
                            : _send,
                        style: FilledButton.styleFrom(
                          shape: const CircleBorder(),
                          padding: const EdgeInsets.all(12),
                        ),
                        child: Icon(
                          c.isGenerating ? Icons.stop : Icons.arrow_upward,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              c.locked
                  ? 'Workspace locked · encrypted data preserved'
                  : 'AI model not installed · Prompts stay on this device',
              style: TextStyle(
                fontSize: 11,
                color: Theme.of(context).hintColor,
              ),
            ),
          ],
        ),
      ),
    ),
  );
  Future<void> _send() async {
    if (input.text.trim().isEmpty || !c.canWrite || c.isBusy) return;
    final text = input.text;
    input.clear();
    setState(() {});
    await c.send(text);
  }

  Future<void> _chatAction(Chat chat, String action) async {
    if (!c.canWrite) return;
    if (action == 'rename') {
      final value = await _prompt('Rename chat', chat.title);
      if (!mounted || value == null) return;
      await c.renameChat(chat.id, value);
    } else if (action == 'pin') {
      await c.togglePin(chat.id);
    } else if (action == 'delete') {
      final confirmed = await _confirmDelete(
        title: 'Delete this chat locally?',
        message:
            '“${chat.title}” and its messages will be removed from this device.',
      );
      if (!mounted || !confirmed) return;
      await c.deleteChat(chat.id);
    } else if (action == 'move') {
      final projectId = await showDialog<String?>(
        context: context,
        builder: (ctx) => SimpleDialog(
          title: const Text('Move to project'),
          children: [
            SimpleDialogOption(
              onPressed: () => Navigator.pop(ctx, ''),
              child: const Text('No project'),
            ),
            ...c.workspace.projects.map(
              (p) => SimpleDialogOption(
                onPressed: () => Navigator.pop(ctx, p.id),
                child: Text(p.name),
              ),
            ),
          ],
        ),
      );
      if (!mounted || projectId == null) return;
      await c.moveChat(chat.id, projectId.isEmpty ? null : projectId);
    }
  }

  Future<void> _newProject() async {
    final v = await _prompt('New project', '');
    if (!mounted) return;
    if (v != null && v.trim().isNotEmpty) await c.createProject(v);
  }

  void _projectMenu(ChatProject p) async {
    final a = await showMenu<String>(
      context: context,
      position: const RelativeRect.fromLTRB(100, 120, 100, 100),
      items: const [
        PopupMenuItem(value: 'rename', child: Text('Rename')),
        PopupMenuItem(value: 'delete', child: Text('Delete project')),
      ],
    );
    if (!mounted || a == null) return;
    if (a == 'rename') {
      final v = await _prompt('Rename project', p.name);
      if (!mounted) return;
      if (v != null) await c.renameProject(p.id, v);
    }
    if (a == 'delete') {
      final confirmed = await _confirmDelete(
        title: 'Delete this project locally?',
        message:
            'The project will be removed. Its chats will stay on this device and become ungrouped.',
      );
      if (!mounted || !confirmed) return;
      await c.deleteProject(p.id);
    }
  }

  Future<bool> _confirmDelete({
    required String title,
    required String message,
  }) async =>
      await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: Text(title),
          content: Text(message),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Delete locally'),
            ),
          ],
        ),
      ) ??
      false;

  Future<String?> _prompt(String title, String initial) async {
    final t = TextEditingController(text: initial);
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title),
        content: TextField(controller: t, autofocus: true),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, t.text),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    t.dispose();
    return result;
  }

  void _themeDialog() => showDialog<void>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: const Text('Appearance'),
      content: DropdownButton<ThemeMode>(
        value: widget.mode,
        items: const [
          DropdownMenuItem(value: ThemeMode.system, child: Text('System')),
          DropdownMenuItem(value: ThemeMode.light, child: Text('Light')),
          DropdownMenuItem(value: ThemeMode.dark, child: Text('Dark')),
        ],
        onChanged: (mode) {
          if (mode != null) widget.onMode(mode);
          Navigator.pop(ctx);
        },
      ),
    ),
  );
  void _info(String key) => showDialog<void>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: Text(
        key == 'help'
            ? 'Help'
            : key == 'learn'
            ? 'Learn more'
            : 'Account',
      ),
      content: Text(
        key == 'account'
            ? 'Sign in is unavailable. Chats are stored encrypted on this device.'
            : 'Legal Chat is a local workspace. Model generation, cloud sync, and share links are not available yet.',
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(ctx),
          child: const Text('Close'),
        ),
      ],
    ),
  );
  void _showShare() => showDialog<void>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: const Text('Share'),
      content: const Text(
        'Choose a response, chat, or project and set read-only or independent continuation permissions when sharing becomes available. Link creation and resharing require the future online service.',
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(ctx),
          child: const Text('Close'),
        ),
      ],
    ),
  );
}
