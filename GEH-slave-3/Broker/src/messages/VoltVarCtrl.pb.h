// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: VoltVarCtrl.proto

#ifndef PROTOBUF_VoltVarCtrl_2eproto__INCLUDED
#define PROTOBUF_VoltVarCtrl_2eproto__INCLUDED

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
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace freedm {
namespace broker {
namespace vvc {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_VoltVarCtrl_2eproto();
void protobuf_AssignDesc_VoltVarCtrl_2eproto();
void protobuf_ShutdownFile_VoltVarCtrl_2eproto();

class VoltageDeltaMessage;
class LineReadingsMessage;
class GradientMessage;
class VoltVarMessage;

// ===================================================================

class VoltageDeltaMessage : public ::google::protobuf::Message {
 public:
  VoltageDeltaMessage();
  virtual ~VoltageDeltaMessage();

  VoltageDeltaMessage(const VoltageDeltaMessage& from);

  inline VoltageDeltaMessage& operator=(const VoltageDeltaMessage& from) {
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
  static const VoltageDeltaMessage& default_instance();

  void Swap(VoltageDeltaMessage* other);

  // implements Message ----------------------------------------------

  VoltageDeltaMessage* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const VoltageDeltaMessage& from);
  void MergeFrom(const VoltageDeltaMessage& from);
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

  // required uint32 control_factor = 1;
  inline bool has_control_factor() const;
  inline void clear_control_factor();
  static const int kControlFactorFieldNumber = 1;
  inline ::google::protobuf::uint32 control_factor() const;
  inline void set_control_factor(::google::protobuf::uint32 value);

  // required float phase_measurement = 2;
  inline bool has_phase_measurement() const;
  inline void clear_phase_measurement();
  static const int kPhaseMeasurementFieldNumber = 2;
  inline float phase_measurement() const;
  inline void set_phase_measurement(float value);

  // optional string reading_location = 3;
  inline bool has_reading_location() const;
  inline void clear_reading_location();
  static const int kReadingLocationFieldNumber = 3;
  inline const ::std::string& reading_location() const;
  inline void set_reading_location(const ::std::string& value);
  inline void set_reading_location(const char* value);
  inline void set_reading_location(const char* value, size_t size);
  inline ::std::string* mutable_reading_location();
  inline ::std::string* release_reading_location();
  inline void set_allocated_reading_location(::std::string* reading_location);

  // @@protoc_insertion_point(class_scope:freedm.broker.vvc.VoltageDeltaMessage)
 private:
  inline void set_has_control_factor();
  inline void clear_has_control_factor();
  inline void set_has_phase_measurement();
  inline void clear_has_phase_measurement();
  inline void set_has_reading_location();
  inline void clear_has_reading_location();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::uint32 control_factor_;
  float phase_measurement_;
  ::std::string* reading_location_;
  friend void  protobuf_AddDesc_VoltVarCtrl_2eproto();
  friend void protobuf_AssignDesc_VoltVarCtrl_2eproto();
  friend void protobuf_ShutdownFile_VoltVarCtrl_2eproto();

  void InitAsDefaultInstance();
  static VoltageDeltaMessage* default_instance_;
};
// -------------------------------------------------------------------

class LineReadingsMessage : public ::google::protobuf::Message {
 public:
  LineReadingsMessage();
  virtual ~LineReadingsMessage();

  LineReadingsMessage(const LineReadingsMessage& from);

  inline LineReadingsMessage& operator=(const LineReadingsMessage& from) {
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
  static const LineReadingsMessage& default_instance();

  void Swap(LineReadingsMessage* other);

  // implements Message ----------------------------------------------

  LineReadingsMessage* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const LineReadingsMessage& from);
  void MergeFrom(const LineReadingsMessage& from);
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

  // repeated float measurement = 1;
  inline int measurement_size() const;
  inline void clear_measurement();
  static const int kMeasurementFieldNumber = 1;
  inline float measurement(int index) const;
  inline void set_measurement(int index, float value);
  inline void add_measurement(float value);
  inline const ::google::protobuf::RepeatedField< float >&
      measurement() const;
  inline ::google::protobuf::RepeatedField< float >*
      mutable_measurement();

  // required string capture_time = 2;
  inline bool has_capture_time() const;
  inline void clear_capture_time();
  static const int kCaptureTimeFieldNumber = 2;
  inline const ::std::string& capture_time() const;
  inline void set_capture_time(const ::std::string& value);
  inline void set_capture_time(const char* value);
  inline void set_capture_time(const char* value, size_t size);
  inline ::std::string* mutable_capture_time();
  inline ::std::string* release_capture_time();
  inline void set_allocated_capture_time(::std::string* capture_time);

  // @@protoc_insertion_point(class_scope:freedm.broker.vvc.LineReadingsMessage)
 private:
  inline void set_has_capture_time();
  inline void clear_has_capture_time();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< float > measurement_;
  ::std::string* capture_time_;
  friend void  protobuf_AddDesc_VoltVarCtrl_2eproto();
  friend void protobuf_AssignDesc_VoltVarCtrl_2eproto();
  friend void protobuf_ShutdownFile_VoltVarCtrl_2eproto();

  void InitAsDefaultInstance();
  static LineReadingsMessage* default_instance_;
};
// -------------------------------------------------------------------

class GradientMessage : public ::google::protobuf::Message {
 public:
  GradientMessage();
  virtual ~GradientMessage();

  GradientMessage(const GradientMessage& from);

  inline GradientMessage& operator=(const GradientMessage& from) {
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
  static const GradientMessage& default_instance();

  void Swap(GradientMessage* other);

  // implements Message ----------------------------------------------

  GradientMessage* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const GradientMessage& from);
  void MergeFrom(const GradientMessage& from);
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

  // repeated double gradient_value = 1;
  inline int gradient_value_size() const;
  inline void clear_gradient_value();
  static const int kGradientValueFieldNumber = 1;
  inline double gradient_value(int index) const;
  inline void set_gradient_value(int index, double value);
  inline void add_gradient_value(double value);
  inline const ::google::protobuf::RepeatedField< double >&
      gradient_value() const;
  inline ::google::protobuf::RepeatedField< double >*
      mutable_gradient_value();

  // required string gradient_capture_time = 2;
  inline bool has_gradient_capture_time() const;
  inline void clear_gradient_capture_time();
  static const int kGradientCaptureTimeFieldNumber = 2;
  inline const ::std::string& gradient_capture_time() const;
  inline void set_gradient_capture_time(const ::std::string& value);
  inline void set_gradient_capture_time(const char* value);
  inline void set_gradient_capture_time(const char* value, size_t size);
  inline ::std::string* mutable_gradient_capture_time();
  inline ::std::string* release_gradient_capture_time();
  inline void set_allocated_gradient_capture_time(::std::string* gradient_capture_time);

  // @@protoc_insertion_point(class_scope:freedm.broker.vvc.GradientMessage)
 private:
  inline void set_has_gradient_capture_time();
  inline void clear_has_gradient_capture_time();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< double > gradient_value_;
  ::std::string* gradient_capture_time_;
  friend void  protobuf_AddDesc_VoltVarCtrl_2eproto();
  friend void protobuf_AssignDesc_VoltVarCtrl_2eproto();
  friend void protobuf_ShutdownFile_VoltVarCtrl_2eproto();

  void InitAsDefaultInstance();
  static GradientMessage* default_instance_;
};
// -------------------------------------------------------------------

class VoltVarMessage : public ::google::protobuf::Message {
 public:
  VoltVarMessage();
  virtual ~VoltVarMessage();

  VoltVarMessage(const VoltVarMessage& from);

  inline VoltVarMessage& operator=(const VoltVarMessage& from) {
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
  static const VoltVarMessage& default_instance();

  void Swap(VoltVarMessage* other);

  // implements Message ----------------------------------------------

  VoltVarMessage* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const VoltVarMessage& from);
  void MergeFrom(const VoltVarMessage& from);
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

  // optional .freedm.broker.vvc.VoltageDeltaMessage voltage_delta_message = 1;
  inline bool has_voltage_delta_message() const;
  inline void clear_voltage_delta_message();
  static const int kVoltageDeltaMessageFieldNumber = 1;
  inline const ::freedm::broker::vvc::VoltageDeltaMessage& voltage_delta_message() const;
  inline ::freedm::broker::vvc::VoltageDeltaMessage* mutable_voltage_delta_message();
  inline ::freedm::broker::vvc::VoltageDeltaMessage* release_voltage_delta_message();
  inline void set_allocated_voltage_delta_message(::freedm::broker::vvc::VoltageDeltaMessage* voltage_delta_message);

  // optional .freedm.broker.vvc.LineReadingsMessage line_readings_message = 2;
  inline bool has_line_readings_message() const;
  inline void clear_line_readings_message();
  static const int kLineReadingsMessageFieldNumber = 2;
  inline const ::freedm::broker::vvc::LineReadingsMessage& line_readings_message() const;
  inline ::freedm::broker::vvc::LineReadingsMessage* mutable_line_readings_message();
  inline ::freedm::broker::vvc::LineReadingsMessage* release_line_readings_message();
  inline void set_allocated_line_readings_message(::freedm::broker::vvc::LineReadingsMessage* line_readings_message);

  // optional .freedm.broker.vvc.GradientMessage gradient_message = 3;
  inline bool has_gradient_message() const;
  inline void clear_gradient_message();
  static const int kGradientMessageFieldNumber = 3;
  inline const ::freedm::broker::vvc::GradientMessage& gradient_message() const;
  inline ::freedm::broker::vvc::GradientMessage* mutable_gradient_message();
  inline ::freedm::broker::vvc::GradientMessage* release_gradient_message();
  inline void set_allocated_gradient_message(::freedm::broker::vvc::GradientMessage* gradient_message);

  // @@protoc_insertion_point(class_scope:freedm.broker.vvc.VoltVarMessage)
 private:
  inline void set_has_voltage_delta_message();
  inline void clear_has_voltage_delta_message();
  inline void set_has_line_readings_message();
  inline void clear_has_line_readings_message();
  inline void set_has_gradient_message();
  inline void clear_has_gradient_message();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::freedm::broker::vvc::VoltageDeltaMessage* voltage_delta_message_;
  ::freedm::broker::vvc::LineReadingsMessage* line_readings_message_;
  ::freedm::broker::vvc::GradientMessage* gradient_message_;
  friend void  protobuf_AddDesc_VoltVarCtrl_2eproto();
  friend void protobuf_AssignDesc_VoltVarCtrl_2eproto();
  friend void protobuf_ShutdownFile_VoltVarCtrl_2eproto();

  void InitAsDefaultInstance();
  static VoltVarMessage* default_instance_;
};
// ===================================================================


// ===================================================================

// VoltageDeltaMessage

// required uint32 control_factor = 1;
inline bool VoltageDeltaMessage::has_control_factor() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void VoltageDeltaMessage::set_has_control_factor() {
  _has_bits_[0] |= 0x00000001u;
}
inline void VoltageDeltaMessage::clear_has_control_factor() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void VoltageDeltaMessage::clear_control_factor() {
  control_factor_ = 0u;
  clear_has_control_factor();
}
inline ::google::protobuf::uint32 VoltageDeltaMessage::control_factor() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.VoltageDeltaMessage.control_factor)
  return control_factor_;
}
inline void VoltageDeltaMessage::set_control_factor(::google::protobuf::uint32 value) {
  set_has_control_factor();
  control_factor_ = value;
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.VoltageDeltaMessage.control_factor)
}

// required float phase_measurement = 2;
inline bool VoltageDeltaMessage::has_phase_measurement() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void VoltageDeltaMessage::set_has_phase_measurement() {
  _has_bits_[0] |= 0x00000002u;
}
inline void VoltageDeltaMessage::clear_has_phase_measurement() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void VoltageDeltaMessage::clear_phase_measurement() {
  phase_measurement_ = 0;
  clear_has_phase_measurement();
}
inline float VoltageDeltaMessage::phase_measurement() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.VoltageDeltaMessage.phase_measurement)
  return phase_measurement_;
}
inline void VoltageDeltaMessage::set_phase_measurement(float value) {
  set_has_phase_measurement();
  phase_measurement_ = value;
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.VoltageDeltaMessage.phase_measurement)
}

// optional string reading_location = 3;
inline bool VoltageDeltaMessage::has_reading_location() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void VoltageDeltaMessage::set_has_reading_location() {
  _has_bits_[0] |= 0x00000004u;
}
inline void VoltageDeltaMessage::clear_has_reading_location() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void VoltageDeltaMessage::clear_reading_location() {
  if (reading_location_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    reading_location_->clear();
  }
  clear_has_reading_location();
}
inline const ::std::string& VoltageDeltaMessage::reading_location() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.VoltageDeltaMessage.reading_location)
  return *reading_location_;
}
inline void VoltageDeltaMessage::set_reading_location(const ::std::string& value) {
  set_has_reading_location();
  if (reading_location_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    reading_location_ = new ::std::string;
  }
  reading_location_->assign(value);
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.VoltageDeltaMessage.reading_location)
}
inline void VoltageDeltaMessage::set_reading_location(const char* value) {
  set_has_reading_location();
  if (reading_location_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    reading_location_ = new ::std::string;
  }
  reading_location_->assign(value);
  // @@protoc_insertion_point(field_set_char:freedm.broker.vvc.VoltageDeltaMessage.reading_location)
}
inline void VoltageDeltaMessage::set_reading_location(const char* value, size_t size) {
  set_has_reading_location();
  if (reading_location_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    reading_location_ = new ::std::string;
  }
  reading_location_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:freedm.broker.vvc.VoltageDeltaMessage.reading_location)
}
inline ::std::string* VoltageDeltaMessage::mutable_reading_location() {
  set_has_reading_location();
  if (reading_location_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    reading_location_ = new ::std::string;
  }
  // @@protoc_insertion_point(field_mutable:freedm.broker.vvc.VoltageDeltaMessage.reading_location)
  return reading_location_;
}
inline ::std::string* VoltageDeltaMessage::release_reading_location() {
  clear_has_reading_location();
  if (reading_location_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    return NULL;
  } else {
    ::std::string* temp = reading_location_;
    reading_location_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    return temp;
  }
}
inline void VoltageDeltaMessage::set_allocated_reading_location(::std::string* reading_location) {
  if (reading_location_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete reading_location_;
  }
  if (reading_location) {
    set_has_reading_location();
    reading_location_ = reading_location;
  } else {
    clear_has_reading_location();
    reading_location_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.vvc.VoltageDeltaMessage.reading_location)
}

// -------------------------------------------------------------------

// LineReadingsMessage

// repeated float measurement = 1;
inline int LineReadingsMessage::measurement_size() const {
  return measurement_.size();
}
inline void LineReadingsMessage::clear_measurement() {
  measurement_.Clear();
}
inline float LineReadingsMessage::measurement(int index) const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.LineReadingsMessage.measurement)
  return measurement_.Get(index);
}
inline void LineReadingsMessage::set_measurement(int index, float value) {
  measurement_.Set(index, value);
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.LineReadingsMessage.measurement)
}
inline void LineReadingsMessage::add_measurement(float value) {
  measurement_.Add(value);
  // @@protoc_insertion_point(field_add:freedm.broker.vvc.LineReadingsMessage.measurement)
}
inline const ::google::protobuf::RepeatedField< float >&
LineReadingsMessage::measurement() const {
  // @@protoc_insertion_point(field_list:freedm.broker.vvc.LineReadingsMessage.measurement)
  return measurement_;
}
inline ::google::protobuf::RepeatedField< float >*
LineReadingsMessage::mutable_measurement() {
  // @@protoc_insertion_point(field_mutable_list:freedm.broker.vvc.LineReadingsMessage.measurement)
  return &measurement_;
}

// required string capture_time = 2;
inline bool LineReadingsMessage::has_capture_time() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void LineReadingsMessage::set_has_capture_time() {
  _has_bits_[0] |= 0x00000002u;
}
inline void LineReadingsMessage::clear_has_capture_time() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void LineReadingsMessage::clear_capture_time() {
  if (capture_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    capture_time_->clear();
  }
  clear_has_capture_time();
}
inline const ::std::string& LineReadingsMessage::capture_time() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.LineReadingsMessage.capture_time)
  return *capture_time_;
}
inline void LineReadingsMessage::set_capture_time(const ::std::string& value) {
  set_has_capture_time();
  if (capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    capture_time_ = new ::std::string;
  }
  capture_time_->assign(value);
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.LineReadingsMessage.capture_time)
}
inline void LineReadingsMessage::set_capture_time(const char* value) {
  set_has_capture_time();
  if (capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    capture_time_ = new ::std::string;
  }
  capture_time_->assign(value);
  // @@protoc_insertion_point(field_set_char:freedm.broker.vvc.LineReadingsMessage.capture_time)
}
inline void LineReadingsMessage::set_capture_time(const char* value, size_t size) {
  set_has_capture_time();
  if (capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    capture_time_ = new ::std::string;
  }
  capture_time_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:freedm.broker.vvc.LineReadingsMessage.capture_time)
}
inline ::std::string* LineReadingsMessage::mutable_capture_time() {
  set_has_capture_time();
  if (capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    capture_time_ = new ::std::string;
  }
  // @@protoc_insertion_point(field_mutable:freedm.broker.vvc.LineReadingsMessage.capture_time)
  return capture_time_;
}
inline ::std::string* LineReadingsMessage::release_capture_time() {
  clear_has_capture_time();
  if (capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    return NULL;
  } else {
    ::std::string* temp = capture_time_;
    capture_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    return temp;
  }
}
inline void LineReadingsMessage::set_allocated_capture_time(::std::string* capture_time) {
  if (capture_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete capture_time_;
  }
  if (capture_time) {
    set_has_capture_time();
    capture_time_ = capture_time;
  } else {
    clear_has_capture_time();
    capture_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.vvc.LineReadingsMessage.capture_time)
}

// -------------------------------------------------------------------

// GradientMessage

// repeated double gradient_value = 1;
inline int GradientMessage::gradient_value_size() const {
  return gradient_value_.size();
}
inline void GradientMessage::clear_gradient_value() {
  gradient_value_.Clear();
}
inline double GradientMessage::gradient_value(int index) const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.GradientMessage.gradient_value)
  return gradient_value_.Get(index);
}
inline void GradientMessage::set_gradient_value(int index, double value) {
  gradient_value_.Set(index, value);
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.GradientMessage.gradient_value)
}
inline void GradientMessage::add_gradient_value(double value) {
  gradient_value_.Add(value);
  // @@protoc_insertion_point(field_add:freedm.broker.vvc.GradientMessage.gradient_value)
}
inline const ::google::protobuf::RepeatedField< double >&
GradientMessage::gradient_value() const {
  // @@protoc_insertion_point(field_list:freedm.broker.vvc.GradientMessage.gradient_value)
  return gradient_value_;
}
inline ::google::protobuf::RepeatedField< double >*
GradientMessage::mutable_gradient_value() {
  // @@protoc_insertion_point(field_mutable_list:freedm.broker.vvc.GradientMessage.gradient_value)
  return &gradient_value_;
}

// required string gradient_capture_time = 2;
inline bool GradientMessage::has_gradient_capture_time() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void GradientMessage::set_has_gradient_capture_time() {
  _has_bits_[0] |= 0x00000002u;
}
inline void GradientMessage::clear_has_gradient_capture_time() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void GradientMessage::clear_gradient_capture_time() {
  if (gradient_capture_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    gradient_capture_time_->clear();
  }
  clear_has_gradient_capture_time();
}
inline const ::std::string& GradientMessage::gradient_capture_time() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.GradientMessage.gradient_capture_time)
  return *gradient_capture_time_;
}
inline void GradientMessage::set_gradient_capture_time(const ::std::string& value) {
  set_has_gradient_capture_time();
  if (gradient_capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    gradient_capture_time_ = new ::std::string;
  }
  gradient_capture_time_->assign(value);
  // @@protoc_insertion_point(field_set:freedm.broker.vvc.GradientMessage.gradient_capture_time)
}
inline void GradientMessage::set_gradient_capture_time(const char* value) {
  set_has_gradient_capture_time();
  if (gradient_capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    gradient_capture_time_ = new ::std::string;
  }
  gradient_capture_time_->assign(value);
  // @@protoc_insertion_point(field_set_char:freedm.broker.vvc.GradientMessage.gradient_capture_time)
}
inline void GradientMessage::set_gradient_capture_time(const char* value, size_t size) {
  set_has_gradient_capture_time();
  if (gradient_capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    gradient_capture_time_ = new ::std::string;
  }
  gradient_capture_time_->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:freedm.broker.vvc.GradientMessage.gradient_capture_time)
}
inline ::std::string* GradientMessage::mutable_gradient_capture_time() {
  set_has_gradient_capture_time();
  if (gradient_capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    gradient_capture_time_ = new ::std::string;
  }
  // @@protoc_insertion_point(field_mutable:freedm.broker.vvc.GradientMessage.gradient_capture_time)
  return gradient_capture_time_;
}
inline ::std::string* GradientMessage::release_gradient_capture_time() {
  clear_has_gradient_capture_time();
  if (gradient_capture_time_ == &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    return NULL;
  } else {
    ::std::string* temp = gradient_capture_time_;
    gradient_capture_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
    return temp;
  }
}
inline void GradientMessage::set_allocated_gradient_capture_time(::std::string* gradient_capture_time) {
  if (gradient_capture_time_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete gradient_capture_time_;
  }
  if (gradient_capture_time) {
    set_has_gradient_capture_time();
    gradient_capture_time_ = gradient_capture_time;
  } else {
    clear_has_gradient_capture_time();
    gradient_capture_time_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.vvc.GradientMessage.gradient_capture_time)
}

// -------------------------------------------------------------------

// VoltVarMessage

// optional .freedm.broker.vvc.VoltageDeltaMessage voltage_delta_message = 1;
inline bool VoltVarMessage::has_voltage_delta_message() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void VoltVarMessage::set_has_voltage_delta_message() {
  _has_bits_[0] |= 0x00000001u;
}
inline void VoltVarMessage::clear_has_voltage_delta_message() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void VoltVarMessage::clear_voltage_delta_message() {
  if (voltage_delta_message_ != NULL) voltage_delta_message_->::freedm::broker::vvc::VoltageDeltaMessage::Clear();
  clear_has_voltage_delta_message();
}
inline const ::freedm::broker::vvc::VoltageDeltaMessage& VoltVarMessage::voltage_delta_message() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.VoltVarMessage.voltage_delta_message)
  return voltage_delta_message_ != NULL ? *voltage_delta_message_ : *default_instance_->voltage_delta_message_;
}
inline ::freedm::broker::vvc::VoltageDeltaMessage* VoltVarMessage::mutable_voltage_delta_message() {
  set_has_voltage_delta_message();
  if (voltage_delta_message_ == NULL) voltage_delta_message_ = new ::freedm::broker::vvc::VoltageDeltaMessage;
  // @@protoc_insertion_point(field_mutable:freedm.broker.vvc.VoltVarMessage.voltage_delta_message)
  return voltage_delta_message_;
}
inline ::freedm::broker::vvc::VoltageDeltaMessage* VoltVarMessage::release_voltage_delta_message() {
  clear_has_voltage_delta_message();
  ::freedm::broker::vvc::VoltageDeltaMessage* temp = voltage_delta_message_;
  voltage_delta_message_ = NULL;
  return temp;
}
inline void VoltVarMessage::set_allocated_voltage_delta_message(::freedm::broker::vvc::VoltageDeltaMessage* voltage_delta_message) {
  delete voltage_delta_message_;
  voltage_delta_message_ = voltage_delta_message;
  if (voltage_delta_message) {
    set_has_voltage_delta_message();
  } else {
    clear_has_voltage_delta_message();
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.vvc.VoltVarMessage.voltage_delta_message)
}

// optional .freedm.broker.vvc.LineReadingsMessage line_readings_message = 2;
inline bool VoltVarMessage::has_line_readings_message() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void VoltVarMessage::set_has_line_readings_message() {
  _has_bits_[0] |= 0x00000002u;
}
inline void VoltVarMessage::clear_has_line_readings_message() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void VoltVarMessage::clear_line_readings_message() {
  if (line_readings_message_ != NULL) line_readings_message_->::freedm::broker::vvc::LineReadingsMessage::Clear();
  clear_has_line_readings_message();
}
inline const ::freedm::broker::vvc::LineReadingsMessage& VoltVarMessage::line_readings_message() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.VoltVarMessage.line_readings_message)
  return line_readings_message_ != NULL ? *line_readings_message_ : *default_instance_->line_readings_message_;
}
inline ::freedm::broker::vvc::LineReadingsMessage* VoltVarMessage::mutable_line_readings_message() {
  set_has_line_readings_message();
  if (line_readings_message_ == NULL) line_readings_message_ = new ::freedm::broker::vvc::LineReadingsMessage;
  // @@protoc_insertion_point(field_mutable:freedm.broker.vvc.VoltVarMessage.line_readings_message)
  return line_readings_message_;
}
inline ::freedm::broker::vvc::LineReadingsMessage* VoltVarMessage::release_line_readings_message() {
  clear_has_line_readings_message();
  ::freedm::broker::vvc::LineReadingsMessage* temp = line_readings_message_;
  line_readings_message_ = NULL;
  return temp;
}
inline void VoltVarMessage::set_allocated_line_readings_message(::freedm::broker::vvc::LineReadingsMessage* line_readings_message) {
  delete line_readings_message_;
  line_readings_message_ = line_readings_message;
  if (line_readings_message) {
    set_has_line_readings_message();
  } else {
    clear_has_line_readings_message();
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.vvc.VoltVarMessage.line_readings_message)
}

// optional .freedm.broker.vvc.GradientMessage gradient_message = 3;
inline bool VoltVarMessage::has_gradient_message() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void VoltVarMessage::set_has_gradient_message() {
  _has_bits_[0] |= 0x00000004u;
}
inline void VoltVarMessage::clear_has_gradient_message() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void VoltVarMessage::clear_gradient_message() {
  if (gradient_message_ != NULL) gradient_message_->::freedm::broker::vvc::GradientMessage::Clear();
  clear_has_gradient_message();
}
inline const ::freedm::broker::vvc::GradientMessage& VoltVarMessage::gradient_message() const {
  // @@protoc_insertion_point(field_get:freedm.broker.vvc.VoltVarMessage.gradient_message)
  return gradient_message_ != NULL ? *gradient_message_ : *default_instance_->gradient_message_;
}
inline ::freedm::broker::vvc::GradientMessage* VoltVarMessage::mutable_gradient_message() {
  set_has_gradient_message();
  if (gradient_message_ == NULL) gradient_message_ = new ::freedm::broker::vvc::GradientMessage;
  // @@protoc_insertion_point(field_mutable:freedm.broker.vvc.VoltVarMessage.gradient_message)
  return gradient_message_;
}
inline ::freedm::broker::vvc::GradientMessage* VoltVarMessage::release_gradient_message() {
  clear_has_gradient_message();
  ::freedm::broker::vvc::GradientMessage* temp = gradient_message_;
  gradient_message_ = NULL;
  return temp;
}
inline void VoltVarMessage::set_allocated_gradient_message(::freedm::broker::vvc::GradientMessage* gradient_message) {
  delete gradient_message_;
  gradient_message_ = gradient_message;
  if (gradient_message) {
    set_has_gradient_message();
  } else {
    clear_has_gradient_message();
  }
  // @@protoc_insertion_point(field_set_allocated:freedm.broker.vvc.VoltVarMessage.gradient_message)
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace vvc
}  // namespace broker
}  // namespace freedm

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_VoltVarCtrl_2eproto__INCLUDED
