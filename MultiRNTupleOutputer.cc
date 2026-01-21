#include <iostream>
#include "MultiRNTupleOutputer.h"
#include "OutputerFactory.h"
#include "FunctorTask.h"
#include "RNTupleOutputerConfig.h"
#include "RNTupleOutputerFieldMaker.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>

using namespace cce::tf;

MultiRNTupleOutputer::MultiRNTupleOutputer(std::string const& fileName, unsigned int iNLanes, RNTupleOutputerConfig const& iConfig):
    fileName_(fileName),
    entries_(iNLanes),
    config_(iConfig),
    collateTime_{std::chrono::microseconds::zero()},
    parallelTime_{0}
  { }

void MultiRNTupleOutputer::setupForLane(unsigned int iLaneIndex, std::vector<DataProductRetriever> const& iDPs) {
  if ( iLaneIndex == 0 ) {
    const std::string eventAuxiliaryBranchName{"EventAuxiliary"}; 
    
    auto writeOptions = writeOptionsFrom(config_);

    RNTupleOutputerFieldMaker fieldMaker(config_);
    std::vector<std::unique_ptr<ROOT::RNTupleModel>> models;
    std::unique_ptr<ROOT::RFieldBase> auxField;

    for(auto const& dp: iDPs) {
      auto model = ROOT::RNTupleModel::CreateBare();

      // chop last . if present
      auto name = dp.name().substr(0, dp.name().find("."));
      if ( config_.verbose_ > 1 ) std::cout << "-------- Creating field for " << name << " of type " << dp.classType()->GetName() << "\n";
      try { 
        auto field = fieldMaker.make(name, dp.classType()->GetName());
        assert(field);
        if ( config_.verbose_ > 1 ) ROOT::Internal::RPrintSchemaVisitor(std::cout, '*', 1000, 10).VisitField(*field);

        if(dp.name() == eventAuxiliaryBranchName) auxField = std::move(field);
        else {
          model->AddField(std::move(field));
          models.push_back(std::move(model));
        }
      }
      catch (ROOT::RException& e) {
         std::cout << "Failed: " << e.what() << "\n";
         throw std::runtime_error("Failed to create field");
      }
    }

    if(not auxField) {
      id_ = std::make_shared<EventIdentifier>();
      auxField = ROOT::RFieldBase::Create("EventID", "cce::tf::EventIdentifier").Unwrap();
      if ( config_.verbose_ > 1 ) ROOT::Internal::RPrintSchemaVisitor(std::cout, '*', 1000, 10).VisitField(*auxField);
      assert(auxField);
    }

    //In DUNE's new framework, every data product has its own RNTuple.  And every RNTuple has its own "index" field.
    //Use the auxiliary field from old framework as a stand-in for "index" field in new framework.
    for(auto& model: models) {
      assert(!model->GetFieldNames().empty());
      assert(model->GetFieldNames().size() == 1);
      auto const& name = *model->GetFieldNames().begin();
      model->AddField(auxField->Clone(auxField->GetFieldName()));
      if(config_.printEstimateWriteMemoryUsage_) {
        std::cout <<"RNTupleWriter: EstimateWriteMemoryUsage "<<model->EstimateWriteMemoryUsage(writeOptions)<<std::endl;
      }
      ntuples_[name] = ROOT::RNTupleWriter::Recreate(std::move(model), "Events_" + name, fileName_, writeOptions);
    }
  }
  else if ( ntuples_.empty() ) {
    throw std::logic_error("setupForLane should be sequential");
  }
  entries_[iLaneIndex].retrievers = &iDPs;
}

void MultiRNTupleOutputer::productReadyAsync(unsigned int iLaneIndex, DataProductRetriever const& iDataProduct, TaskHolder iCallback) const {
}

void MultiRNTupleOutputer::outputAsync(unsigned int iLaneIndex, EventIdentifier const& iEventID, TaskHolder iCallback) const {
  auto start = std::chrono::high_resolution_clock::now();
  auto group = iCallback.group();

  for(auto const& iDP: *entries_[iLaneIndex].retrievers) {
    if(iDP.name().find("EventAuxiliary") == std::string::npos) {
      collateQueue_.push(*group, [this, iEventID, iLaneIndex, &iDP, callback=std::move(iCallback)]() mutable {
          collateProducts(iEventID, iDP, std::move(callback));
        });
    }
  }
  auto time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
  parallelTime_ += time.count();
}

void MultiRNTupleOutputer::printSummary() const {
  auto start = std::chrono::high_resolution_clock::now();
  ntuples_.clear();
  auto deleteTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

  start = std::chrono::high_resolution_clock::now();

  std::cout <<"MultiRNTupleOutputer\n"
    "  total serial collate time at end event: "<<collateTime_.count()<<"us\n"
    "  total non-serializer parallel time at end event: "<<parallelTime_.load()<<"us\n"
    "  end of job RNTupleWriter shutdown time: "<<deleteTime.count()<<"us\n";
}

void MultiRNTupleOutputer::collateProducts(
    EventIdentifier const& iEventID,
    DataProductRetriever const& iDP,
    TaskHolder iCallback
    ) const
{
  auto start = std::chrono::high_resolution_clock::now();
  auto thisOffset = eventGlobalOffset_++;
  if ( config_.verbose_ > 0 ) std::cout << thisOffset << " event id " << iEventID.run << ", "<< iEventID.lumi<<", "<<iEventID.event<<"\n";

  auto const name = iDP.name().substr(0, iDP.name().find("."));
  auto& ntuple = ntuples_[name];
  assert(ntuple);
  auto rentry = ntuple->CreateEntry();
  void** ptr = iDP.address();
  rentry->BindRawPtr(name, *ptr);

  if(id_) {
    *id_ = iEventID;
    rentry->BindRawPtr("EventID", id_.get());
  }
  ntuple->Fill(*rentry);

  collateTime_ += std::chrono::duration_cast<decltype(collateTime_)>(std::chrono::high_resolution_clock::now() - start);
}


namespace {
class Maker : public OutputerMakerBase {
  public:
    Maker(): OutputerMakerBase("MultiRNTupleOutputer") {}
    std::unique_ptr<OutputerBase> create(unsigned int iNLanes, ConfigurationParameters const& params) const final {

      auto result = parseRNTupleConfig(params);
      if(not result) {
        return {};
      }
      return std::make_unique<MultiRNTupleOutputer>(result->first, iNLanes, result->second);
    }
};

Maker s_maker;
}
