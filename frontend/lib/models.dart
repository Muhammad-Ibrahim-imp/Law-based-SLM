import 'dart:convert';

enum MessageRole { user, assistant }

class ChatAttachment {
  const ChatAttachment({
    required this.localPath,
    required this.name,
    required this.mediaType,
  });

  final String localPath;
  final String name;
  final String mediaType;

  Map<String, Object?> toJson() => {
    'localPath': localPath,
    'name': name,
    'mediaType': mediaType,
  };

  factory ChatAttachment.fromJson(Map<String, dynamic> j) => ChatAttachment(
    localPath: j['localPath'],
    name: j['name'],
    mediaType: j['mediaType'],
  );
}

class ChatMessage {
  ChatMessage({
    required this.id,
    required this.role,
    required this.text,
    required this.createdAt,
    List<ChatAttachment>? attachments,
  }) : attachments = attachments ?? [];
  final String id;
  final MessageRole role;
  String text;
  final DateTime createdAt;
  final List<ChatAttachment> attachments;
  Map<String, Object?> toJson() => {
    'id': id,
    'role': role.name,
    'text': text,
    'createdAt': createdAt.toIso8601String(),
    'attachments': attachments.map((a) => a.toJson()).toList(),
  };
  factory ChatMessage.fromJson(Map<String, dynamic> j) => ChatMessage(
    id: j['id'],
    role: MessageRole.values.byName(j['role']),
    text: j['text'],
    createdAt: DateTime.parse(j['createdAt']),
    attachments: (j['attachments'] as List? ?? [])
        .map((a) => ChatAttachment.fromJson(a))
        .toList(),
  );
}

class Chat {
  Chat({
    required this.id,
    required this.title,
    required this.updatedAt,
    this.projectId,
    this.pinned = false,
    List<ChatMessage>? messages,
  }) : messages = messages ?? [];
  final String id;
  String title;
  DateTime updatedAt;
  String? projectId;
  bool pinned;
  final List<ChatMessage> messages;
  Map<String, Object?> toJson() => {
    'id': id,
    'title': title,
    'updatedAt': updatedAt.toIso8601String(),
    'projectId': projectId,
    'pinned': pinned,
    'messages': messages.map((m) => m.toJson()).toList(),
  };
  factory Chat.fromJson(Map<String, dynamic> j) => Chat(
    id: j['id'],
    title: j['title'],
    updatedAt: DateTime.parse(j['updatedAt']),
    projectId: j['projectId'],
    pinned: j['pinned'] ?? false,
    messages: (j['messages'] as List)
        .map((m) => ChatMessage.fromJson(m))
        .toList(),
  );
}

class ChatProject {
  ChatProject({required this.id, required this.name});
  final String id;
  String name;
  Map<String, Object?> toJson() => {'id': id, 'name': name};
  factory ChatProject.fromJson(Map<String, dynamic> j) =>
      ChatProject(id: j['id'], name: j['name']);
}

class Workspace {
  Workspace({
    List<Chat>? chats,
    List<ChatProject>? projects,
    this.themeMode = 'dark',
  }) : chats = chats ?? [],
       projects = projects ?? [];
  final List<Chat> chats;
  final List<ChatProject> projects;
  String themeMode;
  Map<String, Object?> toJson() => {
    'chats': chats.map((c) => c.toJson()).toList(),
    'projects': projects.map((p) => p.toJson()).toList(),
    'themeMode': themeMode,
  };
  factory Workspace.fromJson(Map<String, dynamic> j) => Workspace(
    chats: (j['chats'] as List).map((c) => Chat.fromJson(c)).toList(),
    projects: (j['projects'] as List)
        .map((p) => ChatProject.fromJson(p))
        .toList(),
    themeMode: j['themeMode'] ?? 'dark',
  );
  String encode() => jsonEncode(toJson());
  factory Workspace.decode(String source) =>
      Workspace.fromJson(jsonDecode(source));
}
