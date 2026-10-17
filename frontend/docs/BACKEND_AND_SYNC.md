# Adding models and online services later

## Embedded model replacement

There is no Ollama connection, model download, localhost service, or simulated assistant in this version. The user refined the original Ollama requirement to an embedded, on-device runtime and postponed supplying model files. The UI can save user prompts now; generation remains unavailable.

The model boundary is `lib/backend.dart`. `ChatBackend` exposes availability, attachment capability, role-aware message history, a `Stream<String>` of generated text chunks, and cancellation. The UI calls `ChatController`; it does not call an inference runtime.

To connect the temporary model:

1. Confirm its format, license, chat template, memory needs, and the first physical device's hardware. Choose and verify a compatible embedded runtime for the supported platforms. No runtime dependency is installed yet.
2. Add one implementation, for example `lib/embedded_backend.dart`, implementing `ChatBackend`. Keep native-runtime calls and model loading there. Keep file paths, model identity, and runtime settings together in that implementation or one accompanying configuration file.
3. In `lib/main.dart`, replace the `UnconfiguredBackend()` passed to `ChatController` with that implementation and import its file. `UnconfiguredBackend` can then be deleted without changing message rendering or persistence.
4. Emit text deltas, honor cancellation, and report failures through the stream. Run expensive inference outside the UI isolate. Don't emit the whole accumulated answer as every chunk, since the controller appends each chunk.
5. Update the no-model labels and available model choices in `lib/main.dart`. Add provider tests for load failure, cancellation, context handling, and generation, followed by physical-device tests.

For the real backend, add an implementation that runs the family SLM, criminal SLM, or Jev/Laya router locally. Those are three choices: the router is not a third answer-generating SLM, and both SLMs must not answer the same request. Keep routing and inference inside the backend implementation. Add selected-model identity to requests and actual answering-model identity to messages when that behavior is implemented. The current model files/API do not exist, so no concrete routing contract is invented here.

The current role-aware conversation model is deliberately independent of Ollama and any inference package. Runtime compatibility with the custom trained architectures is not yet established.

## Attachments

`ChatMessage.attachments` stores optional `ChatAttachment` references containing a local path, display name, and media type; old messages without that field still load. `ChatBackend.supportsAttachments` is false in the unconfigured implementation. No files are currently picked, read, or uploaded.

When a backend supports PDFs, Word files, or another format, implement its ingestion/parser, validate file type and size, add encrypted storage for attachment contents, and wire the disabled attachment button in `lib/main.dart`. Enable it only for compatible backends. A capability flag alone does not supply document-reading functionality.

## Accounts and sync

Authentication, servers, and sync are deferred. This version has a local workspace, not a real signed-in account. The account and sharing menus state that those services are unavailable. Local encryption uses a device key; losing that key can make local history unrecoverable.

The intended future behavior is:

- Locally created chats remain on that device and work offline. Sync them with the signed-in account when connected.
- A newly signed-in device needs internet to list account history. Opening a remote-only chat asks whether to save it locally. Either answer leaves a download control available. A downloaded copy can be reopened offline.
- **Remove download** removes only the device copy. **Delete chat** removes it across the account and all devices once sync reaches them. Use deletion tombstones to prevent stale devices from restoring deleted chats.
- Deleting a never-synced offline chat removes it locally and prevents later upload. Deleting a previously synced chat offline must queue a global deletion.
- Projects are named folders of chats. Account isolation, sync conflicts, authentication expiry, and key recovery need a concrete service design before release. Current device keys cannot simply be copied into cloud records to implement encryption across accounts.

## Sharing

The future share service must support a response, entire chat, or project. The sender chooses read-only access or independent continuation, plus whether the recipient may reshare. A recipient's continuation creates their own fork and never changes the sender's history. Links should open the installed app or a web viewer.

Access permissions and resharing restrictions must be checked by that service, including revocation and unauthorized access. There is no server, domain, deep-link registration, web viewer, or working share link in this version. The UI explains this instead of manufacturing a link.
