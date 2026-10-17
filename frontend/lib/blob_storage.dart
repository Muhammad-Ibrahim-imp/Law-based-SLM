abstract interface class BlobStore {
  Future<String?> read();
  Future<void> writeAtomic(String value);
}
