// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ProtocolMessage.proto

#ifndef PROTOBUF_ProtocolMessage_2eproto__INCLUDED
#define PROTOBUF_ProtocolMessage_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2006000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
#include "ModuleMessage.pb.h"
// @@protoc_insertion_point(includes)

namespace freedm {
namespace broker {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_ProtocolMessage_2eproto();
void protobuf_AssignDesc_ProtocolMessage_2eproto();
void protobuf_ShutdownFile_ProtocolMessage_2eproto();

class ProtocolMessage;
class ProtocolMessageWindow;

enum ProtocolMessage_Status {
  ProtocolMessage_Status_CREATED = 1,
  ProtocolMessage_Status_ACCEPTED = 2,
  ProtocolMessage_Status_BAD_REQUEST = 3,
  ProtocolMessage_Status_MESSAGE = 4
};
bool ProtocolMessage_Status_IsValid(int value);
const ProtocolMessage_Status ProtocolMessage_Status_Status_MIN = ProtocolMessage_Status_CREATED;
const ProtocolMessage_Status ProtocolMessage_Status_Status_MAX = ProtocolMessage_Status_MESSAGE;
const int ProtocolMessage_Status_Status_ARRAYSIZE = ProtocolMessage_Status_Status_MAX + 1;

const ::google::protobuf::EnumDescriptor* ProtocolMessage_Status_descriptor();
inline const ::std::string& ProtocolMessage_Status_Name(ProtocolMessage_Status value) {
  return ::google::protobuf::internal::NameOfEnum(
    ProtocolMessage_Status_descriptor(), value);
}
inline bool ProtocolMessage_Status_Parse(
    const ::std::string& name, ProtocolMessage_Status* value) {
  return ::google::protobuf::internal::ParseNamedEnum<ProtocolMessage_Status>(
    ProtocolMessage_Status_descriptor(), name, value);
}
// ===================================================================

class ProtocolMessage : public ::google::protobuf::Message {
 public:
  ProtocolMessage();
  virtual ~ProtocolMessage();

  ProtocolMessage(const ProtocolMessage& from);

  inline ProtocolMessage& operator=(const ProtocolMessage& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const ProtocolMessage& default_instance();

  void Swap(ProtocolMessage* other);

  // implements Message ----------------------------------------------

  ProtocolMessage* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const ProtocolMessage& from);
  void MergeFrom(const ProtocolMessage& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  typedef ProtocolMessage_Status Status;
  static const Status CREATED = ProtocolMessage_Status_CREATED;
  static const Status ACCEPTED = ProtocolMessage_Status_ACCEPTED;
  static const Status BAD_REQUEST = ProtocolMessage_Status_BAD_REQUEST;
  static const Status MESSAGE = ProtocolMessage_Status_MESSAGE;
  static inline bool Status_IsValid(int value) {
    return ProtocolMessage_Status_IsValid(value);
  }
  static const Status Status_MIN =
    ProtocolMessage_Status_Status_MIN;
  static const Status Status_MAX =
    ProtocolMessage_Status_Status_MAX;
  static const int Status_ARRAYSIZE =
    ProtocolMessage_Status_Status_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  Status_descriptor() {
    return ProtocolMessage_Status_descriptor();
  }
  static inline const ::std::string& Status_Name(Status value) {
    return ProtocolMessage_Status_Name(value);
  }
  static inline bool Status_Parse(const ::std::string& name,
      Status* value) {
    return ProtocolMessage_Status_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // required uint32 sequence_num = 3;
  inline bool has_sequence_num() const;
  inline void clear_sequence_num();
  static const int kSequenceNumFieldNumber = 3;
  inline ::google::protobuf::uint32 sequence_num() const;
  inline void set_sequence_num(::google::protobuf::uint32 value);

  // optional string expire_time = 4;
  inline bool has_expire_time() const;
  inline void clear_expire_time();
  static const int kExpireTimeFieldNumber = 4;
  inline const ::std::string& expire_time() const;
  inline void set_expire_time(const ::std::string& value);
  inline void set_expire_time(const char* value);
  inline void set_expire_time(const char* value, size_t size);
  inline ::std::string* mutable_expire_time();
  inline ::std::string* release_expire_time();
  inline void set_allocated_expire_time(::std::string* expire_time);

  // required .freedm.broker.ProtocolMessage.Status status = 5;
  inline bool has_status() const;
  inline void clear_status();
  static const int kStatusFieldNumber = 5;
  inline ::freedm::broker::ProtocolMessage_Status status() const;
  inline void set_status(::freedm::broker::ProtocolMessage_Status value);

  // optional int32 kill = 6;
  inline bool has_kill() const;
  inline void clear_kill();
  static const int kKillFieldNumber = 6;
  inline ::google::protobuf::int32 kill() const;
  inline void set_kill(::google::protobuf::int32 value);

  // optional fixed64 hash = 7;
  inline bool has_hash() const;
  inline void clear_hash();
  static const int kHashFieldNumber = 7;
  inline ::google::protobuf::uint64 hash() const;
  inline void set_hash(::google::protobuf::uint64 value);

  // optional .freedm.broker.ModuleMessage module_message = 8;
  inline bool has_module_message() const;
  inline void clear_module_message();
  static const int kModuleMessageFieldNumber = 8;
  inline const ::freedm::broker::ModuleMessage& module_message() const;
  inline ::freedm::broker::ModuleMessage* mutable_module_message();
  inline ::freedm::broker::ModuleMessage* release_module_message();
  inline void set_allocated_module_message(::freedm::broker::ModuleMessage* module_message);

  // @@protoc_insertion_point(class_scope:freedm.broker.ProtocolMessage)
 private:
  inline void set_has_sequence_num();
  inline void clear_has_sequence_num();
  inline void set_has_expire_time();
  inline void clear_has_expire_time();
  inline void set_has_status();
  inline void clear_has_status();
  inline void set_has_kill();
  inline void clear_has_kill();
  inline void set_has_hash();
  inline void clear_has_hash();
  inline void set_has_module_message();
  inline void clear_has_module_message();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::std::string* expire_time_;
  ::google::protobuf::uint32 sequence_num_;
  int status_;
  ::google::protobuf::uint64 hash_;
  ::freedm::broker::ModuleMessage* module_message_;
  ::google::protobuf::int32 kill_;
  friend void  protobuf_AddDesc_ProtocolMessage_2eproto();
  friend void protobuf_AssignDesc_ProtocolMessage_2eproto();
  friend void protobuf_ShutdownFile_ProtocolMessage_2eproto();

  void InitAsDefaultInstance();
  static ProtocolMessage* default_instance_;
};
// -------------------------------------------------------------------

class ProtocolMessageWindow : public ::google::protobuf::Message {
 public:
  ProtocolMessageWindow();
  virtual ~ProtocolMessageWindow();

  ProtocolMessageWindow(const ProtocolMessageWindow& from);

  inline ProtocolMessageWindow& operator=(const ProtocolMessageWindow& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const ProtocolMessageWindow& default_instance();

  void Swap(ProtocolMessageWindow* other);

  // implements Message ----------------------------------------------

  ProtocolMessageWindow* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const ProtocolMessageWindow& from);
  void MergeFrom(const ProtocolMessageWindow& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required string source_uuid = 1;
  inline bool has_source_uuid() const;
  inline void clear_source_uuid();
  static const int kSourceUuidFieldNumber = 1;
  inline const ::std::string& source_uuid() const;
  inline void set_source_uuid(const ::std::string& value);
  inline void set_source_uuid(const char* value);
  inline void set_source_uuid(const char* value, size_t size);
  inline ::std::string* mutable_source_uuid();
  inline ::std::string* release_source_uuid();
  inline void set_allocated_source_uuid(::std::string* source_uuid);

  // required string send_time = 2;
  inline bool has_send_time() const;
  inline void clear_send_time();
  static const int kSendTimeFieldNumber = 2;
  inline const ::std::string& send_time() const;
  inline void set_send_time(const ::std::string& value);
  inline void set_send_time(const char* value);
  inline void set_send_time(const char* value, size_t size);
  inline ::std::string* mutable_send_time();
  inline ::std::string* release_send_time();
  inline void set_allocated_send_time(::std::string* send_time);

  // repeated .freedm.broker.ProtocolMessage messages = 3;
  inline int messages_size() const;
  inline void clear_messages();
  static const int kMessagesFieldNumber = 3;
  inline const ::freedm::broker::ProtocolMessage& messages(int index) const;
  inline ::freedm::broker::ProtocolMessage* mutable_messages(int index);
  inline ::freedm::broker::ProtocolMessage* add_messages();
  inline const ::google::protobuf::RepeatedPtrField< ::freedm::broker::ProtocolMessage >&
      messages() const;
  inline ::google::protobuf::RepeatedPtrField< ::freedm::broker::ProtocolMessage >*
      mutable_messages();

  // @@protoc_insertion_point(class_scope:freedm.broker.ProtocolMessageWindow)
 private:
  inline void set_has_source_uuid();
  inline void clear_has_source_uuid();
  inline void set_has_send_time();
  inline void clear_has_send_time();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::std::string* source_uuid_;
  ::std::string* send_time_;
  ::google::protobuf::RepeatedPtrField< ::freedm::broker::ProtocolMessage > messages_;
  friend void  protobuf_AddDesc_ProtocolMessage_2eproto();
  friend void protobuf_AssignDesc_ProtocolMessage_2eproto();
  friend void protobuf_ShutdownFile_ProtocolMessage_2eproto();

  void InitAsDefaultInstance();
  static ProtocolMessageWindow* default_instance_;
};
// ===================================================================


// ===================================================================

// ProtocolMessage

// required uint32 sequence_num = 3;
inline bool ProtocolMessage::has_sequence_num() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void ProtocolMessage::set_has_sequence_num() {
  _has_bits_[0] |= 0x00000001u;
}
inline void ProtocolMessage::clear_has_sequence_num() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void ProtocolMessage::clear_sequence_num() {
  sequence_num_ = 0u;
  clear_has_sequence_num();
}
inline ::google::protobuf::uint32 ProtocolMessage::sequence_num() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessage.sequence_num)
  return sequence_num_;
}
inline void ProtocolMessage::set_sequence_num(::google::protobuf::uint32 value) {
  set_has_sequence_num();
  sequence_num_ = value;
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessage.sequence_num)
}

// optional string expire_time = 4;
inline bool ProtocolMessage::has_expire_time() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void ProtocolMessage::set_has_expire_time() {
  _has_bits_[0] |= 0x00000002u;
}
inline void ProtocolMessage::clear_has_expire_time() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void ProtocolMessage::clear_expire_time() {
  if (expire_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    expire_time_->clear();
  }
  clear_has_expire_time();
}
inline const ::std::string& ProtocolMessage::expire_time() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessage.expire_time)
  return *expire_time_;
}
inline void ProtocolMessage::set_expire_time(const ::std::string& value) {
  set_has_expire_time();
  if (expire_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    expire_time_ = new ::std::string;
  }
  expire_time_->assign(value);
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessage.expire_time)
}
inline void ProtocolMessage::set_expire_time(const char* value) {
  set_has_expire_time();
  if (expire_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    expire_time_ = new ::std::string;
  }
  expire_time_->assign(value);
  // @@protoc_insertion_point(field_set_char:freedm.broker.ProtocolMessage.expire_time)
}
inline void ProtocolMessage::set_expire_time(const char* value, size_t size) {
  set_has_expire_time();
  if (expire_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    expire_time_ = new ::std::string;
  }
  expire_time_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:freedm.broker.ProtocolMessage.expire_time)
}
inline ::std::string* ProtocolMessage::mutable_expire_time() {
  set_has_expire_time();
  if (expire_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    expire_time_ = new ::std::string;
  }
  // @@protoc_insertion_point(field_mutable:freedm.broker.ProtocolMessage.expire_time)
  return expire_time_;
}
inline ::std::string* ProtocolMessage::release_expire_time() {
  clear_has_expire_time();
  if (expire_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    return NULL;
  } else {
    ::std::string* temp = expire_time_;
    expire_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    return temp;
  }
}
inline void ProtocolMessage::set_allocated_expire_time(::std::string* expire_time) {
  if (expire_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete expire_time_;
  }
  if (expire_time) {
    set_has_expire_time();
    expire_time_ = expire_time;
  } else {
    clear_has_expire_time();
    expire_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.ProtocolMessage.expire_time)
}

// required .freedm.broker.ProtocolMessage.Status status = 5;
inline bool ProtocolMessage::has_status() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void ProtocolMessage::set_has_status() {
  _has_bits_[0] |= 0x00000004u;
}
inline void ProtocolMessage::clear_has_status() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void ProtocolMessage::clear_status() {
  status_ = 1;
  clear_has_status();
}
inline ::freedm::broker::ProtocolMessage_Status ProtocolMessage::status() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessage.status)
  return static_cast< ::freedm::broker::ProtocolMessage_Status >(status_);
}
inline void ProtocolMessage::set_status(::freedm::broker::ProtocolMessage_Status value) {
  assert(::freedm::broker::ProtocolMessage_Status_IsValid(value));
  set_has_status();
  status_ = value;
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessage.status)
}

// optional int32 kill = 6;
inline bool ProtocolMessage::has_kill() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void ProtocolMessage::set_has_kill() {
  _has_bits_[0] |= 0x00000008u;
}
inline void ProtocolMessage::clear_has_kill() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void ProtocolMessage::clear_kill() {
  kill_ = 0;
  clear_has_kill();
}
inline ::google::protobuf::int32 ProtocolMessage::kill() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessage.kill)
  return kill_;
}
inline void ProtocolMessage::set_kill(::google::protobuf::int32 value) {
  set_has_kill();
  kill_ = value;
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessage.kill)
}

// optional fixed64 hash = 7;
inline bool ProtocolMessage::has_hash() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void ProtocolMessage::set_has_hash() {
  _has_bits_[0] |= 0x00000010u;
}
inline void ProtocolMessage::clear_has_hash() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void ProtocolMessage::clear_hash() {
  hash_ = GOOGLE_ULONGLONG(0);
  clear_has_hash();
}
inline ::google::protobuf::uint64 ProtocolMessage::hash() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessage.hash)
  return hash_;
}
inline void ProtocolMessage::set_hash(::google::protobuf::uint64 value) {
  set_has_hash();
  hash_ = value;
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessage.hash)
}

// optional .freedm.broker.ModuleMessage module_message = 8;
inline bool ProtocolMessage::has_module_message() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void ProtocolMessage::set_has_module_message() {
  _has_bits_[0] |= 0x00000020u;
}
inline void ProtocolMessage::clear_has_module_message() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void ProtocolMessage::clear_module_message() {
  if (module_message_ != NULL) module_message_->::freedm::broker::ModuleMessage::Clear();
  clear_has_module_message();
}
inline const ::freedm::broker::ModuleMessage& ProtocolMessage::module_message() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessage.module_message)
  return module_message_ != NULL ? *module_message_ : *default_instance_->module_message_;
}
inline ::freedm::broker::ModuleMessage* ProtocolMessage::mutable_module_message() {
  set_has_module_message();
  if (module_message_ == NULL) module_message_ = new ::freedm::broker::ModuleMessage;
  // @@protoc_insertion_point(field_mutable:freedm.broker.ProtocolMessage.module_message)
  return module_message_;
}
inline ::freedm::broker::ModuleMessage* ProtocolMessage::release_module_message() {
  clear_has_module_message();
  ::freedm::broker::ModuleMessage* temp = module_message_;
  module_message_ = NULL;
  return temp;
}
inline void ProtocolMessage::set_allocated_module_message(::freedm::broker::ModuleMessage* module_message) {
  delete module_message_;
  module_message_ = module_message;
  if (module_message) {
    set_has_module_message();
  } else {
    clear_has_module_message();
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.ProtocolMessage.module_message)
}

// -------------------------------------------------------------------

// ProtocolMessageWindow

// required string source_uuid = 1;
inline bool ProtocolMessageWindow::has_source_uuid() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void ProtocolMessageWindow::set_has_source_uuid() {
  _has_bits_[0] |= 0x00000001u;
}
inline void ProtocolMessageWindow::clear_has_source_uuid() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void ProtocolMessageWindow::clear_source_uuid() {
  if (source_uuid_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    source_uuid_->clear();
  }
  clear_has_source_uuid();
}
inline const ::std::string& ProtocolMessageWindow::source_uuid() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessageWindow.source_uuid)
  return *source_uuid_;
}
inline void ProtocolMessageWindow::set_source_uuid(const ::std::string& value) {
  set_has_source_uuid();
  if (source_uuid_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    source_uuid_ = new ::std::string;
  }
  source_uuid_->assign(value);
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessageWindow.source_uuid)
}
inline void ProtocolMessageWindow::set_source_uuid(const char* value) {
  set_has_source_uuid();
  if (source_uuid_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    source_uuid_ = new ::std::string;
  }
  source_uuid_->assign(value);
  // @@protoc_insertion_point(field_set_char:freedm.broker.ProtocolMessageWindow.source_uuid)
}
inline void ProtocolMessageWindow::set_source_uuid(const char* value, size_t size) {
  set_has_source_uuid();
  if (source_uuid_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    source_uuid_ = new ::std::string;
  }
  source_uuid_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:freedm.broker.ProtocolMessageWindow.source_uuid)
}
inline ::std::string* ProtocolMessageWindow::mutable_source_uuid() {
  set_has_source_uuid();
  if (source_uuid_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    source_uuid_ = new ::std::string;
  }
  // @@protoc_insertion_point(field_mutable:freedm.broker.ProtocolMessageWindow.source_uuid)
  return source_uuid_;
}
inline ::std::string* ProtocolMessageWindow::release_source_uuid() {
  clear_has_source_uuid();
  if (source_uuid_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    return NULL;
  } else {
    ::std::string* temp = source_uuid_;
    source_uuid_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    return temp;
  }
}
inline void ProtocolMessageWindow::set_allocated_source_uuid(::std::string* source_uuid) {
  if (source_uuid_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete source_uuid_;
  }
  if (source_uuid) {
    set_has_source_uuid();
    source_uuid_ = source_uuid;
  } else {
    clear_has_source_uuid();
    source_uuid_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.ProtocolMessageWindow.source_uuid)
}

// required string send_time = 2;
inline bool ProtocolMessageWindow::has_send_time() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void ProtocolMessageWindow::set_has_send_time() {
  _has_bits_[0] |= 0x00000002u;
}
inline void ProtocolMessageWindow::clear_has_send_time() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void ProtocolMessageWindow::clear_send_time() {
  if (send_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    send_time_->clear();
  }
  clear_has_send_time();
}
inline const ::std::string& ProtocolMessageWindow::send_time() const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessageWindow.send_time)
  return *send_time_;
}
inline void ProtocolMessageWindow::set_send_time(const ::std::string& value) {
  set_has_send_time();
  if (send_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    send_time_ = new ::std::string;
  }
  send_time_->assign(value);
  // @@protoc_insertion_point(field_set:freedm.broker.ProtocolMessageWindow.send_time)
}
inline void ProtocolMessageWindow::set_send_time(const char* value) {
  set_has_send_time();
  if (send_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    send_time_ = new ::std::string;
  }
  send_time_->assign(value);
  // @@protoc_insertion_point(field_set_char:freedm.broker.ProtocolMessageWindow.send_time)
}
inline void ProtocolMessageWindow::set_send_time(const char* value, size_t size) {
  set_has_send_time();
  if (send_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    send_time_ = new ::std::string;
  }
  send_time_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:freedm.broker.ProtocolMessageWindow.send_time)
}
inline ::std::string* ProtocolMessageWindow::mutable_send_time() {
  set_has_send_time();
  if (send_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    send_time_ = new ::std::string;
  }
  // @@protoc_insertion_point(field_mutable:freedm.broker.ProtocolMessageWindow.send_time)
  return send_time_;
}
inline ::std::string* ProtocolMessageWindow::release_send_time() {
  clear_has_send_time();
  if (send_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    return NULL;
  } else {
    ::std::string* temp = send_time_;
    send_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    return temp;
  }
}
inline void ProtocolMessageWindow::set_allocated_send_time(::std::string* send_time) {
  if (send_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete send_time_;
  }
  if (send_time) {
    set_has_send_time();
    send_time_ = send_time;
  } else {
    clear_has_send_time();
    send_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.ProtocolMessageWindow.send_time)
}

// repeated .freedm.broker.ProtocolMessage messages = 3;
inline int ProtocolMessageWindow::messages_size() const {
  return messages_.size();
}
inline void ProtocolMessageWindow::clear_messages() {
  messages_.Clear();
}
inline const ::freedm::broker::ProtocolMessage& ProtocolMessageWindow::messages(int index) const {
  // @@protoc_insertion_point(field_get:freedm.broker.ProtocolMessageWindow.messages)
  return messages_.Get(index);
}
inline ::freedm::broker::ProtocolMessage* ProtocolMessageWindow::mutable_messages(int index) {
  // @@protoc_insertion_point(field_mutable:freedm.broker.ProtocolMessageWindow.messages)
  return messages_.Mutable(index);
}
inline ::freedm::broker::ProtocolMessage* ProtocolMessageWindow::add_messages() {
  // @@protoc_insertion_point(field_add:freedm.broker.ProtocolMessageWindow.messages)
  return messages_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::freedm::broker::ProtocolMessage >&
ProtocolMessageWindow::messages() const {
  // @@protoc_insertion_point(field_list:freedm.broker.ProtocolMessageWindow.messages)
  return messages_;
}
inline ::google::protobuf::RepeatedPtrField< ::freedm::broker::ProtocolMessage >*
ProtocolMessageWindow::mutable_messages() {
  // @@protoc_insertion_point(field_mutable_list:freedm.broker.ProtocolMessageWindow.messages)
  return &messages_;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace broker
}  // namespace freedm

#ifndef SWIG
namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::freedm::broker::ProtocolMessage_Status> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::freedm::broker::ProtocolMessage_Status>() {
  return ::freedm::broker::ProtocolMessage_Status_descriptor();
}

}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_ProtocolMessage_2eproto__INCLUDED
