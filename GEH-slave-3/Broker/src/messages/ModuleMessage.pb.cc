// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ModuleMessage.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "ModuleMessage.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace freedm {
namespace broker {

namespace {

const ::google::protobuf::Descriptor* ModuleMessage_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  ModuleMessage_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_ModuleMessage_2eproto() {
  protobuf_AddDesc_ModuleMessage_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "ModuleMessage.proto");
  GOOGLE_CHECK(file != NULL);
  ModuleMessage_descriptor_ = file->message_type(0);
  static const int ModuleMessage_offsets_[6] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, recipient_module_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, group_management_message_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, state_collection_message_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, load_balancing_message_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, clock_synchronizer_message_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, volt_var_message_),
  };
  ModuleMessage_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      ModuleMessage_descriptor_,
      ModuleMessage::default_instance_,
      ModuleMessage_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(ModuleMessage, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(ModuleMessage));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_ModuleMessage_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    ModuleMessage_descriptor_, &ModuleMessage::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_ModuleMessage_2eproto() {
  delete ModuleMessage::default_instance_;
  delete ModuleMessage_reflection_;
}

void protobuf_AddDesc_ModuleMessage_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::freedm::broker::protobuf_AddDesc_ClockSynchronizer_2eproto();
  ::freedm::broker::gm::protobuf_AddDesc_GroupManagement_2eproto();
  ::freedm::broker::lb::protobuf_AddDesc_LoadBalancing_2eproto();
  ::freedm::broker::sc::protobuf_AddDesc_StateCollection_2eproto();
  ::freedm::broker::vvc::protobuf_AddDesc_VoltVarCtrl_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\023ModuleMessage.proto\022\rfreedm.broker\032\027Cl"
    "ockSynchronizer.proto\032\025GroupManagement.p"
    "roto\032\023LoadBalancing.proto\032\025StateCollecti"
    "on.proto\032\021VoltVarCtrl.proto\"\223\003\n\rModuleMe"
    "ssage\022\030\n\020recipient_module\030\001 \002(\t\022J\n\030group"
    "_management_message\030\002 \001(\0132(.freedm.broke"
    "r.gm.GroupManagementMessage\022J\n\030state_col"
    "lection_message\030\003 \001(\0132(.freedm.broker.sc"
    ".StateCollectionMessage\022F\n\026load_balancin"
    "g_message\030\004 \001(\0132&.freedm.broker.lb.LoadB"
    "alancingMessage\022K\n\032clock_synchronizer_me"
    "ssage\030\005 \001(\0132\'.freedm.broker.ClockSynchro"
    "nizerMessage\022;\n\020volt_var_message\030\006 \001(\0132!"
    ".freedm.broker.vvc.VoltVarMessage", 553);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "ModuleMessage.proto", &protobuf_RegisterTypes);
  ModuleMessage::default_instance_ = new ModuleMessage();
  ModuleMessage::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_ModuleMessage_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_ModuleMessage_2eproto {
  StaticDescriptorInitializer_ModuleMessage_2eproto() {
    protobuf_AddDesc_ModuleMessage_2eproto();
  }
} static_descriptor_initializer_ModuleMessage_2eproto_;

// ===================================================================

#ifndef _MSC_VER
const int ModuleMessage::kRecipientModuleFieldNumber;
const int ModuleMessage::kGroupManagementMessageFieldNumber;
const int ModuleMessage::kStateCollectionMessageFieldNumber;
const int ModuleMessage::kLoadBalancingMessageFieldNumber;
const int ModuleMessage::kClockSynchronizerMessageFieldNumber;
const int ModuleMessage::kVoltVarMessageFieldNumber;
#endif  // !_MSC_VER

ModuleMessage::ModuleMessage()
  : ::google::protobuf::Message() {
  SharedCtor();
  // @@protoc_insertion_point(constructor:freedm.broker.ModuleMessage)
}

void ModuleMessage::InitAsDefaultInstance() {
  group_management_message_ = const_cast< ::freedm::broker::gm::GroupManagementMessage*>(&::freedm::broker::gm::GroupManagementMessage::default_instance());
  state_collection_message_ = const_cast< ::freedm::broker::sc::StateCollectionMessage*>(&::freedm::broker::sc::StateCollectionMessage::default_instance());
  load_balancing_message_ = const_cast< ::freedm::broker::lb::LoadBalancingMessage*>(&::freedm::broker::lb::LoadBalancingMessage::default_instance());
  clock_synchronizer_message_ = const_cast< ::freedm::broker::ClockSynchronizerMessage*>(&::freedm::broker::ClockSynchronizerMessage::default_instance());
  volt_var_message_ = const_cast< ::freedm::broker::vvc::VoltVarMessage*>(&::freedm::broker::vvc::VoltVarMessage::default_instance());
}

ModuleMessage::ModuleMessage(const ModuleMessage& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:freedm.broker.ModuleMessage)
}

void ModuleMessage::SharedCtor() {
  ::google::protobuf::internal::GetEmptyString();
  _cached_size_ = 0;
  recipient_module_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  group_management_message_ = NULL;
  state_collection_message_ = NULL;
  load_balancing_message_ = NULL;
  clock_synchronizer_message_ = NULL;
  volt_var_message_ = NULL;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

ModuleMessage::~ModuleMessage() {
  // @@protoc_insertion_point(destructor:freedm.broker.ModuleMessage)
  SharedDtor();
}

void ModuleMessage::SharedDtor() {
  if (recipient_module_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete recipient_module_;
  }
  if (this != default_instance_) {
    delete group_management_message_;
    delete state_collection_message_;
    delete load_balancing_message_;
    delete clock_synchronizer_message_;
    delete volt_var_message_;
  }
}

void ModuleMessage::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* ModuleMessage::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return ModuleMessage_descriptor_;
}

const ModuleMessage& ModuleMessage::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_ModuleMessage_2eproto();
  return *default_instance_;
}

ModuleMessage* ModuleMessage::default_instance_ = NULL;

ModuleMessage* ModuleMessage::New() const {
  return new ModuleMessage;
}

void ModuleMessage::Clear() {
  if (_has_bits_[0 / 32] & 63) {
    if (has_recipient_module()) {
      if (recipient_module_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
        recipient_module_->clear();
      }
    }
    if (has_group_management_message()) {
      if (group_management_message_ != NULL) group_management_message_->::freedm::broker::gm::GroupManagementMessage::Clear();
    }
    if (has_state_collection_message()) {
      if (state_collection_message_ != NULL) state_collection_message_->::freedm::broker::sc::StateCollectionMessage::Clear();
    }
    if (has_load_balancing_message()) {
      if (load_balancing_message_ != NULL) load_balancing_message_->::freedm::broker::lb::LoadBalancingMessage::Clear();
    }
    if (has_clock_synchronizer_message()) {
      if (clock_synchronizer_message_ != NULL) clock_synchronizer_message_->::freedm::broker::ClockSynchronizerMessage::Clear();
    }
    if (has_volt_var_message()) {
      if (volt_var_message_ != NULL) volt_var_message_->::freedm::broker::vvc::VoltVarMessage::Clear();
    }
  }
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool ModuleMessage::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:freedm.broker.ModuleMessage)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required string recipient_module = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_recipient_module()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
            this->recipient_module().data(), this->recipient_module().length(),
            ::google::protobuf::internal::WireFormat::PARSE,
            "recipient_module");
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_group_management_message;
        break;
      }

      // optional .freedm.broker.gm.GroupManagementMessage group_management_message = 2;
      case 2: {
        if (tag == 18) {
         parse_group_management_message:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_group_management_message()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(26)) goto parse_state_collection_message;
        break;
      }

      // optional .freedm.broker.sc.StateCollectionMessage state_collection_message = 3;
      case 3: {
        if (tag == 26) {
         parse_state_collection_message:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_state_collection_message()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(34)) goto parse_load_balancing_message;
        break;
      }

      // optional .freedm.broker.lb.LoadBalancingMessage load_balancing_message = 4;
      case 4: {
        if (tag == 34) {
         parse_load_balancing_message:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_load_balancing_message()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(42)) goto parse_clock_synchronizer_message;
        break;
      }

      // optional .freedm.broker.ClockSynchronizerMessage clock_synchronizer_message = 5;
      case 5: {
        if (tag == 42) {
         parse_clock_synchronizer_message:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_clock_synchronizer_message()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(50)) goto parse_volt_var_message;
        break;
      }

      // optional .freedm.broker.vvc.VoltVarMessage volt_var_message = 6;
      case 6: {
        if (tag == 50) {
         parse_volt_var_message:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_volt_var_message()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:freedm.broker.ModuleMessage)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:freedm.broker.ModuleMessage)
  return false;
#undef DO_
}

void ModuleMessage::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:freedm.broker.ModuleMessage)
  // required string recipient_module = 1;
  if (has_recipient_module()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->recipient_module().data(), this->recipient_module().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "recipient_module");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->recipient_module(), output);
  }

  // optional .freedm.broker.gm.GroupManagementMessage group_management_message = 2;
  if (has_group_management_message()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, this->group_management_message(), output);
  }

  // optional .freedm.broker.sc.StateCollectionMessage state_collection_message = 3;
  if (has_state_collection_message()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      3, this->state_collection_message(), output);
  }

  // optional .freedm.broker.lb.LoadBalancingMessage load_balancing_message = 4;
  if (has_load_balancing_message()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      4, this->load_balancing_message(), output);
  }

  // optional .freedm.broker.ClockSynchronizerMessage clock_synchronizer_message = 5;
  if (has_clock_synchronizer_message()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      5, this->clock_synchronizer_message(), output);
  }

  // optional .freedm.broker.vvc.VoltVarMessage volt_var_message = 6;
  if (has_volt_var_message()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      6, this->volt_var_message(), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:freedm.broker.ModuleMessage)
}

::google::protobuf::uint8* ModuleMessage::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:freedm.broker.ModuleMessage)
  // required string recipient_module = 1;
  if (has_recipient_module()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->recipient_module().data(), this->recipient_module().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "recipient_module");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->recipient_module(), target);
  }

  // optional .freedm.broker.gm.GroupManagementMessage group_management_message = 2;
  if (has_group_management_message()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        2, this->group_management_message(), target);
  }

  // optional .freedm.broker.sc.StateCollectionMessage state_collection_message = 3;
  if (has_state_collection_message()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        3, this->state_collection_message(), target);
  }

  // optional .freedm.broker.lb.LoadBalancingMessage load_balancing_message = 4;
  if (has_load_balancing_message()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        4, this->load_balancing_message(), target);
  }

  // optional .freedm.broker.ClockSynchronizerMessage clock_synchronizer_message = 5;
  if (has_clock_synchronizer_message()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        5, this->clock_synchronizer_message(), target);
  }

  // optional .freedm.broker.vvc.VoltVarMessage volt_var_message = 6;
  if (has_volt_var_message()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        6, this->volt_var_message(), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:freedm.broker.ModuleMessage)
  return target;
}

int ModuleMessage::ByteSize() const {
  int total_size = 0;

  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required string recipient_module = 1;
    if (has_recipient_module()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->recipient_module());
    }

    // optional .freedm.broker.gm.GroupManagementMessage group_management_message = 2;
    if (has_group_management_message()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->group_management_message());
    }

    // optional .freedm.broker.sc.StateCollectionMessage state_collection_message = 3;
    if (has_state_collection_message()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->state_collection_message());
    }

    // optional .freedm.broker.lb.LoadBalancingMessage load_balancing_message = 4;
    if (has_load_balancing_message()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->load_balancing_message());
    }

    // optional .freedm.broker.ClockSynchronizerMessage clock_synchronizer_message = 5;
    if (has_clock_synchronizer_message()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->clock_synchronizer_message());
    }

    // optional .freedm.broker.vvc.VoltVarMessage volt_var_message = 6;
    if (has_volt_var_message()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->volt_var_message());
    }

  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void ModuleMessage::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const ModuleMessage* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const ModuleMessage*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void ModuleMessage::MergeFrom(const ModuleMessage& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_recipient_module()) {
      set_recipient_module(from.recipient_module());
    }
    if (from.has_group_management_message()) {
      mutable_group_management_message()->::freedm::broker::gm::GroupManagementMessage::MergeFrom(from.group_management_message());
    }
    if (from.has_state_collection_message()) {
      mutable_state_collection_message()->::freedm::broker::sc::StateCollectionMessage::MergeFrom(from.state_collection_message());
    }
    if (from.has_load_balancing_message()) {
      mutable_load_balancing_message()->::freedm::broker::lb::LoadBalancingMessage::MergeFrom(from.load_balancing_message());
    }
    if (from.has_clock_synchronizer_message()) {
      mutable_clock_synchronizer_message()->::freedm::broker::ClockSynchronizerMessage::MergeFrom(from.clock_synchronizer_message());
    }
    if (from.has_volt_var_message()) {
      mutable_volt_var_message()->::freedm::broker::vvc::VoltVarMessage::MergeFrom(from.volt_var_message());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void ModuleMessage::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ModuleMessage::CopyFrom(const ModuleMessage& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ModuleMessage::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000001) != 0x00000001) return false;

  if (has_group_management_message()) {
    if (!this->group_management_message().IsInitialized()) return false;
  }
  if (has_state_collection_message()) {
    if (!this->state_collection_message().IsInitialized()) return false;
  }
  if (has_load_balancing_message()) {
    if (!this->load_balancing_message().IsInitialized()) return false;
  }
  if (has_clock_synchronizer_message()) {
    if (!this->clock_synchronizer_message().IsInitialized()) return false;
  }
  if (has_volt_var_message()) {
    if (!this->volt_var_message().IsInitialized()) return false;
  }
  return true;
}

void ModuleMessage::Swap(ModuleMessage* other) {
  if (other != this) {
    std::swap(recipient_module_, other->recipient_module_);
    std::swap(group_management_message_, other->group_management_message_);
    std::swap(state_collection_message_, other->state_collection_message_);
    std::swap(load_balancing_message_, other->load_balancing_message_);
    std::swap(clock_synchronizer_message_, other->clock_synchronizer_message_);
    std::swap(volt_var_message_, other->volt_var_message_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata ModuleMessage::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = ModuleMessage_descriptor_;
  metadata.reflection = ModuleMessage_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace broker
}  // namespace freedm

// @@protoc_insertion_point(global_scope)
