import 'blob_storage.dart';

class BlobStorage implements BlobStore {
  @override
  Future<String?> read() async =>
      throw UnsupportedError('This platform has no local workspace storage');
  @override
  Future<void> writeAtomic(String value) async =>
      throw UnsupportedError('This platform has no local workspace storage');
}
