
#include "VoltVarCtrl.hpp"

#include <iostream>
#include <fstream>
#include "CLogger.hpp"
#include "Messages.hpp"
#include "CTimings.hpp"
#include "CDeviceManager.hpp"
#include "CGlobalPeerList.hpp"
#include "gm/GroupManagement.hpp"
#include "CGlobalConfiguration.hpp"

#include <sstream>

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/asio/error.hpp>
#include <boost/system/error_code.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <armadillo>

namespace freedm {
	
namespace broker {
		
namespace vvc { 

namespace {
/// This file's logger.
CLocalLogger Logger(__FILE__);
}

///////////////////////////////////////////////////////////////////////////////
/// VVCAgent
/// @description: Constructor for the VVC module
/// @pre: Posix Main should register read handler and invoke this module
/// @post: Object is initialized and ready to run 
/// @param uuid_: This object's uuid
/// @param broker: The Broker
/// @limitations: None
///////////////////////////////////////////////////////////////////////////////
VVCAgent::VVCAgent()
    : ROUND_TIME(boost::posix_time::milliseconds(CTimings::Get("LB_ROUND_TIME")))
    , REQUEST_TIMEOUT(boost::posix_time::milliseconds(CTimings::Get("LB_REQUEST_TIMEOUT")))
{

  Logger.Trace << __PRETTY_FUNCTION__ << std::endl;

  m_RoundTimer = CBroker::Instance().AllocateTimer("vvc");
  m_WaitTimer = CBroker::Instance().AllocateTimer("vvc");
  Pload_vector.zeros(3,1);
  m_Synchronized = 0;

}
VVCAgent::~VVCAgent()
{
}
			
////////////////////////////////////////////////////////////
/// Run
/// @description Main function which initiates the algorithm
/// @pre: Posix Main should invoke this function
/// @post: Triggers the vvc algorithm by calling VVCManage()
/// @limitations None
/////////////////////////////////////////////////////////
int VVCAgent::Run()
{	
  //float sst_1 = device::CDeviceManager::Instance().GetNetValue("LVSST", "AOUT/0");
  //std::cout<< "Real Power got from LVSST is " << sst_1 << std::endl;
  //float sst_10 = device::CDeviceManager::Instance().GetNetValue("LVSST", "AOUT/1");
  //std::cout<< "Reactive Power got from LVSST is " << sst_10 << std::endl;
  std::cout<< " --------------------VVC ---------------------------------" << std::endl; 
  CBroker::Instance().Schedule("vvc",
      boost::bind(&VVCAgent::FirstRound, this, boost::system::error_code()));
  Logger.Info << "VVC is scheduled for the next phase." << std::endl;
  return 0;
}


///////////////////////////////////////////////////////////////////////////////
/// HandleIncomingMessage
/// "Downcasts" incoming messages into a specific message type, and passes the
/// message to an appropriate handler.
/// @pre None
/// @post The message is handled by the target handler or a warning is
///     produced.
/// @param m the incoming message
/// @param peer the node that sent this message (could be this DGI)
///////////////////////////////////////////////////////////////////////////////
void VVCAgent::HandleIncomingMessage(boost::shared_ptr<const ModuleMessage> m, CPeerNode peer)
{
    if(m->has_volt_var_message())
    {
        VoltVarMessage vvm = m->volt_var_message();
        if(vvm.has_voltage_delta_message())
        {
            HandleVoltageDelta(vvm.voltage_delta_message(), peer);
        }
        else if(vvm.has_line_readings_message())
        {
            HandleLineReadings(vvm.line_readings_message(), peer);
        }
	else if(vvm.has_gradient_message())
        {
            HandleGradient(vvm.gradient_message(), peer);
        }
        else
        {
            Logger.Warn << "Dropped unexpected volt var message: \n" << m->DebugString();
        }
    }
    else if(m->has_group_management_message())
    {
        gm::GroupManagementMessage gmm = m->group_management_message();
        if(gmm.has_peer_list_message())
        {
            HandlePeerList(gmm.peer_list_message(), peer);
        }
        else
        {
            Logger.Warn << "Dropped unexpected group management message:\n" << m->DebugString();
        }
    }
    else
    {
        Logger.Warn<<"Dropped message of unexpected type:\n" << m->DebugString();
    }
}

void VVCAgent::HandleVoltageDelta(const VoltageDeltaMessage & m, CPeerNode peer)
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;
    Logger.Notice << "Got VoltageDelta from: " << peer.GetUUID() << std::endl;
    Logger.Notice << "CF "<<m.control_factor()<<" Phase "<<m.phase_measurement()<<std::endl;
}

void VVCAgent::HandleLineReadings(const LineReadingsMessage & m, CPeerNode peer)
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;
    Logger.Notice << "Got Line Readings from "<< peer.GetUUID() << std::endl;
}

void VVCAgent::HandleGradient(const GradientMessage & m, CPeerNode peer)
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;
    Logger.Notice << "Got Vector from "<< peer.GetUUID() << std::endl;
    Logger.Notice << "size of vector "<< m.gradient_value_size() << std::endl;
    using namespace arma;
    //Pload_vector = zeros(3,1);
	
	double x;
	//double x1;
	//double x2;
	//double x3;
	x = m.gradient_value(0); 
    //Pload_vector = zeros(m.gradient_value_size(),1);
	cout << "Gradient Value is: " << m.gradient_value(0) << " from " << peer.GetUUID() << endl;
	//Pload_vector.set_size(3,1); 
    if (peer.GetUUID() == "barbaro:5002")
{
	//x1 = m.gradient_value(0);	
	//Pload_vector(0,0) = x1;
	Pload_vector(0,0) = x;
	cout << "Pload from 5002: " << Pload_vector(0,0) << endl;
}
    else if (peer.GetUUID() == "gloming:5003")
{
	//x2 = m.gradient_value(0);
	//Pload_vector(1,0) = x2;
	Pload_vector(1,0) = x;
	cout << "Pload from 5003: " << Pload_vector(1,0) << endl;
}
    else if (peer.GetUUID() == "arkle:5004")
{
	//x3 = m.gradient_value(0);
	//Pload_vector(2,0) = x3;
	Pload_vector(2,0) = x;
	cout << "Pload from 5004: " << Pload_vector(2,0) << endl;
}
    /*for (int i = 0; i < m.gradient_value_size(); i++)
    {
	  Pload_vector(i,0) = m.gradient_value(i);
    }*/

	cout << "Load Data Received: " << endl << Pload_vector << endl;
    
	//Pload_vector.save("Pload_vec.mat");// must be ROOT to execute
    
}

// HandlePeerlist Implementation
void VVCAgent::HandlePeerList(const gm::PeerListMessage & m, CPeerNode peer)
{
	Logger.Trace << __PRETTY_FUNCTION__ << std::endl;
	Logger.Notice << "Updated Peer List Received from: " << peer.GetUUID() << std::endl;
	
	m_peers.clear();
	
	m_peers = gm::GMAgent::ProcessPeerList(m);
	m_leader = peer.GetUUID();
}

// Preparing Messages
ModuleMessage VVCAgent::VoltageDelta(unsigned int cf, float pm, std::string loc)
{
	VoltVarMessage vvm;
	VoltageDeltaMessage *vdm = vvm.mutable_voltage_delta_message();
	vdm -> set_control_factor(cf);
	vdm -> set_phase_measurement(pm);
	vdm -> set_reading_location(loc);
	return PrepareForSending(vvm,"vvc");
}

ModuleMessage VVCAgent::LineReadings(std::vector<float> vals)
{
	VoltVarMessage vvm;
	std::vector<float>::iterator it;
	LineReadingsMessage *lrm = vvm.mutable_line_readings_message();
	for (it = vals.begin(); it != vals.end(); it++)
	{
		lrm -> add_measurement(*it);
	}
	lrm->set_capture_time(boost::posix_time::to_simple_string(boost::posix_time::microsec_clock::universal_time()));
	return PrepareForSending(vvm,"vvc");
}

ModuleMessage VVCAgent::Gradient(arma::mat grad)
{
	VoltVarMessage vvm;
	unsigned int idx;
        GradientMessage *grdm = vvm.mutable_gradient_message();
	for (idx = 0; idx < grad.n_rows; idx++)
	{
		grdm -> add_gradient_value(grad(idx));
	}
	grdm->set_gradient_capture_time(boost::posix_time::to_simple_string(boost::posix_time::microsec_clock::universal_time()));
	return PrepareForSending(vvm,"vvc");
}


ModuleMessage VVCAgent::PrepareForSending(const VoltVarMessage& message, std::string recipient)
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;
    ModuleMessage mm;
    mm.mutable_volt_var_message()->CopyFrom(message);
    mm.set_recipient_module(recipient);
    return mm;
}
// End of Preparing Messages

///////////////////////////////////////////////////////////////////////////////
/// FirstRound
/// @description The code that is executed as part of the first VVC
///     each round.
/// @pre None
/// @post if the timer wasn't cancelled this function calls the first VVC.
/// @param error The reason this function was called.
///////////////////////////////////////////////////////////////////////////////
void VVCAgent::FirstRound(const boost::system::error_code& err)
{
  Logger.Trace << __PRETTY_FUNCTION__ << std::endl;
  
  if(!err)
  {
    CBroker::Instance().Schedule("vvc", 
	boost::bind(&VVCAgent::VVCManage, this, boost::system::error_code()));
  }
  else if(err == boost::asio::error::operation_aborted)
  {
    Logger.Notice << "VVCManage Aborted" << std::endl;
  }
  else
  {
    Logger.Error << err << std::endl;
    throw boost::system::system_error(err);
  }
			
}


////////////////////////////////////////////////////////////
/// VVCManage
/// @description: Manages the execution of the VVC algorithm 
/// @pre: 
/// @post: 
/// @peers 
/// @limitations
/////////////////////////////////////////////////////////
void VVCAgent::VVCManage(const boost::system::error_code& err)
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;

    if(!err)
    {
      ScheduleNextRound();
      ReadDevices();
      vvc_main();
     
      
    }
        
    else if(err == boost::asio::error::operation_aborted)
    {
        Logger.Notice << "VVCManage Aborted" << std::endl;
    }
    else
    {
        Logger.Error << err << std::endl;
        throw boost::system::system_error(err);
    }
}

///////////////////////////////////////////////////////////////////////////////
/// ScheduleNextRound
/// @description Computes how much time is remaining and if there isn't enough
///     requests the VVC that will run next round.
/// @pre None
/// @post VVCManage is scheduled for this round OR FirstRound is scheduled
///     for next time.
///////////////////////////////////////////////////////////////////////////////
void VVCAgent::ScheduleNextRound()
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;

    if(CBroker::Instance().TimeRemaining() > ROUND_TIME + ROUND_TIME)
    {
        CBroker::Instance().Schedule(m_RoundTimer, ROUND_TIME,
            boost::bind(&VVCAgent::VVCManage, this, boost::asio::placeholders::error));
        Logger.Info << "VVCManage scheduled in " << ROUND_TIME << " ms." << std::endl;
    }
    else
    {
        CBroker::Instance().Schedule(m_RoundTimer, boost::posix_time::not_a_date_time,
            boost::bind(&VVCAgent::FirstRound, this, boost::asio::placeholders::error));
        Logger.Info << "VVCManage scheduled for the next phase." << std::endl;
    }
}


///////////////////////////////////////////////////////////////////////////////
/// ReadDevices
/// @description Reads the device state and updates the appropriate member vars.
/// @pre None
/// @post 
///////////////////////////////////////////////////////////////////////////////
void VVCAgent::ReadDevices()
{
    Logger.Trace << __PRETTY_FUNCTION__ << std::endl;

    float generation = device::CDeviceManager::Instance().GetNetValue("Drer", "generation");
    float storage = device::CDeviceManager::Instance().GetNetValue("Desd", "storage");
    float load = device::CDeviceManager::Instance().GetNetValue("Load", "drain");

    m_Gateway = device::CDeviceManager::Instance().GetNetValue("Sst", "gateway");
    m_NetGeneration = generation + storage - load;
}







void VVCAgent::vvc_main()
{
using namespace arma;
using namespace std;
	
//Prepare para for DPF
sysdata sysinfo = load_system_data();
int Ldl = sysinfo.Dl.n_rows;
int Wdl = sysinfo.Dl.n_cols;
cout << "Dl dimension:"<< Ldl <<"*"<< Wdl << endl;//Matrix Dl in Matlab
mat Dl = sysinfo.Dl;
cx_mat Z = sysinfo.Z;

mat du, step_size; // delta control
double Ploss_aftter_ctrl;
bool flag = true;
double Vmax, Vmin;

//document the original node number in sequence
int j, ja, jb, jc, ia, ib, ic;

j = 1;
ja = 0;
jb = 0;
jc = 0;

int cnt_nodes = 0;
int Lla = 0;
int Llb = 0;
int Llc = 0;
for (int i = 0; i < Ldl; ++i)
{
  if ((int)Dl(i, 0) != 0)
  cnt_nodes = cnt_nodes + 1;
  if ((int)Dl(i, 6) != 0)
    Lla = Lla + 1;
  if ((int)Dl(i, 8) != 0)
    Llb = Llb + 1;
  if ((int)Dl(i, 10) != 0)
    Llc = Llc + 1;
}
cnt_nodes = cnt_nodes + 1;//No of nodes= No of branches +1
mat Node_f = zeros(1,cnt_nodes);
mat Load_a = zeros(1, Lla);
mat Load_b = zeros(1, Llb);
mat Load_c = zeros(1, Llc);

for (int i = 0; i < Ldl && j<cnt_nodes && ja<Lla && jb<Llb &&jc<Llc; ++i)
{
  if (	(int)Dl(i, 2) != 0  )
  {
    Node_f(0, j) = Dl(i, 2);
    ++j;
  }
  if ((int)Dl(i, 6) != 0)
  {
    Load_a(ja) = Dl(i, 2);
    ++ja;
  }
  if ((int)Dl(i, 8) != 0)
  {
    Load_b(jb) = Dl(i, 2);
    ++jb;
  }
  if ((int)Dl(i, 10) != 0)
  {
    Load_c(jc) = Dl(i, 2);
    ++jc;
  }
}
Node_f = Node_f.st();

y_re Y_return = form_Y_abc(Dl, sysinfo.Z, sysinfo.bkva, sysinfo.bkv);
cout << "No. of Branches:" << Y_return.Lnum(0,0) << endl;
cout << "No. of Nodes:" << Y_return.Nnum << endl;
	
int Lbr = Y_return.brnches.n_rows;
int Wbr = Y_return.brnches.n_cols;

//save three phase branch data separately
cx_mat brn_a = cx_mat(zeros(Y_return.Lnum_a, Y_return.brnches.n_cols), zeros(Y_return.Lnum_a, Y_return.brnches.n_cols));
cx_mat brn_b = cx_mat(zeros(Y_return.Lnum_b, Y_return.brnches.n_cols), zeros(Y_return.Lnum_b, Y_return.brnches.n_cols));
cx_mat brn_c = cx_mat(zeros(Y_return.Lnum_c, Y_return.brnches.n_cols), zeros(Y_return.Lnum_c, Y_return.brnches.n_cols));
				
ja = 0;
jb = 0;
jc = 0;

for (int i = 0; i < Lbr && ja<Y_return.Lnum_a && jb<Y_return.Lnum_b && jc<Y_return.Lnum_c; ++i)
{
  if (abs(Y_return.brnches(i, 2)) != 0)
  {
    brn_a.row(ja) = Y_return.brnches.row(i);
    ++ja;
  }
  if (abs(Y_return.brnches(i, 3)) != 0)
  {
    brn_b.row(jb) = Y_return.brnches.row(i);
    ++jb;
  }
  if (abs(Y_return.brnches(i, 4)) != 0)
  {
    brn_c.row(jc) = Y_return.brnches.row(i);
    ++jc;
  }
}


// reading P load
//mat Pload_mat = Pload_vector;
//std::cout << Pload_vector.st() << std::endl;
//bool status = Pload_vector.load("Pload_vec.mat");
//if (status == true)
  //{
	  std::cout << "Pload vector from Slave VVO ... SSTI SSTII SSTIII ..." << std::endl;
	  std::cout << Pload_vector.st() << std::endl;
	  //Dl(0,6) = Pload_vector.at(0,0); //uncomment this
	  //Dl(1,6) = Pload_vector.at(1,0); //uncomment this
	  //Dl(2,6) = Pload_vector.at(2,0); //uncomment this
	  //Dl(0,6) = Pload_mat(0,0); //uncomment this
	  //Dl(1,6) = Pload_mat(1,0); //uncomment this
	  //Dl(2,6) = Pload_mat(2,0); //uncomment this
	  Dl(0,6) = Pload_vector(0,0); //uncomment this
	  Dl(1,6) = Pload_vector(1,0); //uncomment this
	  Dl(2,6) = Pload_vector(2,0); //uncomment this
		
	  std::cout << "Dl(0,6): " << Dl(0,6) << "\nDl(1,6): " << Dl(1,6) << "\nDl(2,6): " << Dl(2,6) << std::endl;
	  //Dl(0,6) = 9; //comment this
	  //Dl(1,6) = 5; //comment this
	  //Dl(2,6) = -3; //comment this
	  Dl.col(8) = Dl.col(6);
	  Dl.col(10) = Dl.col(6);
  //}
  //else
  //{
	//  std::cout <<" Pload from slave VVO not received ... keep default Pload ... "<<std::endl;
  //}


VPQ dpf_re = DPF_return3(Dl, Z);
//cout << "Vpolar = \n" << dpf_re.Vpolar << endl;//comment this
//cout << "PQb = \n" << dpf_re.Vpolar << endl;
mat Vpolar = dpf_re.Vpolar;
mat PQb = dpf_re.PQb;
mat PQL = dpf_re.PQL;

int Lvp = Vpolar.n_rows;
int Wvp = Vpolar.n_cols;
int Lpq = PQb.n_rows;
int Wpq = PQb.n_cols;
double Pla_t = sum(PQL.col(0));
double Plb_t = sum(PQL.col(2));
double Plc_t = sum(PQL.col(4));

// calculate power loss
mat Pload_total, Ploss_orig_ph;
double Ploss_orig;
Pload_total<< Pla_t << Plb_t << Plc_t << endr;
Ploss_orig_ph << (PQb(0, 0) - Pla_t) << (PQb(0, 2) - Plb_t) << (PQb(0, 4) - Plc_t) << endr;
Ploss_orig = accu(Ploss_orig_ph);
//cout << "total load (kW) per phase:" << Pload_total << endl;
cout << "total loss (kW):" << Ploss_orig << endl;



//document the valid Vs and corrsponding node number for each phase
int Lnum_a = Y_return.Lnum_a;
int Lnum_b = Y_return.Lnum_b;
int Lnum_c = Y_return.Lnum_c;

Vabc V_abc = V_abc_list(Vpolar,Node_f,Lvp,Lnum_a,Lnum_b,Lnum_c);

int Lna = V_abc.Lna;
int Lnb = V_abc.Lnb;
int Lnc = V_abc.Lnc;
mat Node_a = V_abc.Node_a;
mat Node_b = V_abc.Node_b;
mat Node_c = V_abc.Node_c;
mat V_a = V_abc.V_a;
mat V_b = V_abc.V_b;
mat V_c = V_abc.V_c;
mat theta_a = V_abc.theta_a;
mat theta_b = V_abc.theta_b;
mat theta_c = V_abc.theta_c;

//Qset before update
mat Qset_a = dpf_re.Qset_a;
mat Qset_b = dpf_re.Qset_b;
mat Qset_c = dpf_re.Qset_c;
mat Dl_new = Dl;
Dl_new.col(7) = Qset_a;
Dl_new.col(9) = Qset_b;
Dl_new.col(11) = Qset_c;

newbrn Newbrn_return = rename_brn(Node_a, Node_b, Node_c, brn_a, brn_b, brn_c, Y_return.Lnum_a, Y_return.Lnum_b, Y_return.Lnum_c, Lna, Lnb, Lnc);

//cout << "Phase A Vmin:" << min(V_a) << endl;
//cout << "Phase B Vmin:" << min(V_b) << endl;
//cout << "Phase C Vmin:" << min(V_c) << endl;
mat Vmin_abc, Vmax_abc;
Vmin_abc << min(min(V_a)) << min(min(V_b)) << min(min(V_c))<<endr;
Vmax_abc << max(max(V_a)) << max(max(V_b)) << max(max(V_c))<<endr;
double Vmin_orig = min(min(Vmin_abc));
double Vmax_orig = max(max(Vmax_abc));
cout << "Vmax (p.u.) = " << Vmax_orig << endl;
cout << "Vmin (p.u.) = " << Vmin_orig << endl;




// ********************************* phase II: Power Loss Minimization **********************************//
// power loss minimization while at each iteration determine Qlimit
// obj F = sum of line losses

// Gradient Calculation

double beta0 = 0.1;// min dQsst for SST is 0.1 kVar
double alpha = 1.1;
int m_max = 100; // max iterations to search for the best step-size
//deltaF/deltaTheta
arma::mat Ftheta_a = form_Ftheta(Y_return.Y_a, V_a, theta_a, Newbrn_return.newbrn_a, Lna, Lnum_a);
arma::mat Ftheta_b = form_Ftheta(Y_return.Y_b, V_b, theta_b, Newbrn_return.newbrn_b, Lnb, Lnum_b);
arma::mat Ftheta_c = form_Ftheta(Y_return.Y_c, V_c, theta_c, Newbrn_return.newbrn_c, Lnc, Lnum_c);

//detltaF/deltaV
arma::mat Fv_a = form_Fv(Y_return.Y_a, V_a, theta_a, Newbrn_return.newbrn_a, Lna, Lnum_a);
arma::mat Fv_b = form_Fv(Y_return.Y_b, V_b, theta_b, Newbrn_return.newbrn_b, Lnb, Lnum_b);
arma::mat Fv_c = form_Fv(Y_return.Y_c, V_c, theta_c, Newbrn_return.newbrn_c, Lnc, Lnum_c);
//detltaF/deltaX
arma::mat Fx_a = join_cols(Ftheta_a, Fv_a);
arma::mat Fx_b = join_cols(Ftheta_b, Fv_b);
arma::mat Fx_c = join_cols(Ftheta_c, Fv_c);
					

//form Jacobian Matrices for each phase
arma::mat J_a, J_b, J_c;
J_a = form_J(Y_return.Y_a, V_a, theta_a, Lna);
J_b = form_J(Y_return.Y_b, V_b, theta_b, Lnb);
J_c = form_J(Y_return.Y_c, V_c, theta_c, Lnc);

// get lambda for each phase
arma::mat lambda_a = -inv(J_a.st())*Fx_a;//need LAPAC etc
arma::mat lambda_b = -inv(J_b.st())*Fx_b;
arma::mat lambda_c = -inv(J_c.st())*Fx_c;

//std::cout << lambda_a << std::endl;
					
//deltaG/delta_Qnj from sst
//deltaP/delta_Qinj
arma::mat Gpq_a = arma::zeros(Lnum_a, Lla);
arma::mat Gpq_b = arma::zeros(Lnum_b, Llb);
arma::mat Gpq_c = arma::zeros(Lnum_c, Llc);
//deltaQ/delta_Qinj
arma::mat Gqq_a = arma::zeros(Lnum_a, Lla);
arma::mat Gqq_b = arma::zeros(Lnum_b, Llb);
arma::mat Gqq_c = arma::zeros(Lnum_c, Llc);
					
					
					//Phase A
					for (ia = 0; ia < Lnum_a; ++ia)
					{
						for ( ja = 0; ja < Lla; ++ja)
						{
							if (Node_a(0, ia + 1) == Load_a(0, ja))
							{
								Gqq_a(ia, ja) = -1;
							}

						}
					}
					
					//Phase B
					for (ib = 0; ib < Lnum_b; ++ib)
					{
						for (jb = 0; jb < Llb; ++jb)
						{
							if (Node_b(0, ib + 1) == Load_b(0, jb))
							{
								Gqq_b(ib, jb) = -1;
							}

						}
					}
					
					//Phase C
					for (ic = 0; ic < Lnum_c; ++ic)
					{
						for (jc = 0; jc < Llc; ++jc)
						{
							if (Node_c(0, ic + 1) == Load_c(0, jc))
							{
								Gqq_c(ic, jc) = -1;
							}

						}
					}
					
//form deltaG/deltaQinj
arma::mat gu_a, gu_b, gu_c;
gu_a = join_cols(Gpq_a, Gqq_a);
gu_b = join_cols(Gpq_b, Gqq_b);
gu_c = join_cols(Gpq_c, Gqq_c);

					

arma::mat g_vq_a = -gu_a.st()*lambda_a;//Gradient in p.u.  aka df/du , u is Qinj
arma::mat g_vq_b = -gu_b.st()*lambda_b;
arma::mat g_vq_c = -gu_c.st()*lambda_c;
//std::cout<< g_vq_a << std::endl;

arma::mat g_min;
g_min << min(min(abs(g_vq_a))) << min(min(abs(g_vq_b))) << min(min(abs(g_vq_c))) << arma::endr;
arma::mat g_max;
g_max << max(max(abs(g_vq_a))) << max(max(abs(g_vq_b))) << max(max(abs(g_vq_c))) << arma::endr;
double gmax = max(max(g_max));
double gmin = min(min(g_min));
std::cout << "max gradient=" << gmax << "	" << "min grad=" << gmin << "(p.u.)" << std::endl;
double gabs_min = min(min(abs( join_cols( g_vq_a, join_cols( g_vq_b, g_vq_c )) )));
// end of gradient calculation	

//cout << "gabs_min=" << gabs_min << endl;
double cvq_a = beta0/(sysinfo.bkva/3)/gabs_min;
double cvq_b = beta0/(sysinfo.bkva/3)/gabs_min;
double cvq_c = beta0/(sysinfo.bkva/3)/gabs_min;

mat ctrl_o =  Dl;// save the previous control


for ( int m = 0; m < m_max; m++ )
{
  // update Qinj for three phase separately
  double Gupdate;
  //Phase A
  for (ia = 0; ia < Lla; ++ia)
  {
    for (ja = 0; ja < Ldl; ++ja)
    {
      if (Dl(ja, 2) == Load_a(0, ia))
      {
	Gupdate = g_vq_a(ia, 0)*(sysinfo.bkva/3)*cvq_a;
	Dl_new(ja, 7) = ctrl_o(ja, 7) - Gupdate;
      }
     }
  }//end of Phase A
  
  //Phase B
  for (ib = 0; ib < Llb; ++ib)
  {
    for (jb = 0; jb < Ldl; ++jb)
    {
      if (Dl(jb, 2) == Load_b(0, ib))
      {
	Gupdate = g_vq_b(ib, 0)*(sysinfo.bkva / 3)*cvq_b;
	Dl_new(jb, 9) = ctrl_o(jb, 9) - Gupdate;
      }
    }
   }//end of Phase B
   
   //Phase C
  for (ic = 0; ic < Llc; ++ic)
  {
    for (jc = 0; jc < Ldl; ++jc)
    {
      if (Dl(jc, 2) == Load_c(0, ic))
      {
	Gupdate = g_vq_c(ic, 0)*(sysinfo.bkva / 3)*cvq_c;
	Dl_new(jc, 11) = ctrl_o(jc, 11) - Gupdate;
      }
     }
  }// end of Phase C
mat Dl_osize = Dl_new;  
mat du_temp=Dl_osize-ctrl_o;
du = du_temp.cols(6,11);

// cout << "Dl_new = \n" << Dl_new << endl;
// DPF based on Dl_new

VPQ dpf_re = DPF_return3(Dl_osize, Z);
//cout << "Vpolar = \n" << dpf_re.Vpolar << endl;
mat Vpolar = dpf_re.Vpolar;
mat PQb = dpf_re.PQb;
mat PQL = dpf_re.PQL;

int Lvp = Vpolar.n_rows;
int Wvp = Vpolar.n_cols;
int Lpq = PQb.n_rows;
int Wpq = PQb.n_cols;
double Pla_t = sum(PQL.col(0));
double Plb_t = sum(PQL.col(2));
double Plc_t = sum(PQL.col(4));

// calculate power loss
mat Pload_total, Ploss_oph;
double Ploss_osize;
Pload_total<< Pla_t << Plb_t << Plc_t << endr;
Ploss_oph << (PQb(0, 0) - Pla_t) << (PQb(0, 2) - Plb_t) << (PQb(0, 4) - Plc_t) << endr;
Ploss_osize = accu(Ploss_oph);
//cout << "total load (kW) per phase:" << Pload_total << endl;
//cout << "total loss (kW):" << Ploss_osize << endl;

Vabc V_abc = V_abc_list(Vpolar,Node_f,Lvp,Lnum_a,Lnum_b,Lnum_c);


mat V_a = V_abc.V_a;
mat V_b = V_abc.V_b;
mat V_c = V_abc.V_c;

mat Vmin_abc, Vmax_abc;
Vmin_abc << min(min(V_a)) << min(min(V_b)) << min(min(V_c))<<endr;
Vmax_abc << max(max(V_a)) << max(max(V_b)) << max(max(V_c))<<endr;
Vmin = min(min(Vmin_abc));
Vmax = max(max(Vmax_abc));


step_size << cvq_a << cvq_b << cvq_c << endr;
//cout << "\n \n step-size at the " << m+1 << "th iteration = " << step_size(0,0) << endl;

// new step-size
cvq_a=alpha*cvq_a;
cvq_b=alpha*cvq_b;
cvq_c=alpha*cvq_c; 

//update Dl_new based on the new step-size
//Phase A
  for (ia = 0; ia < Lla; ++ia)
  {
    for (ja = 0; ja < Ldl; ++ja)
    {
      if (Dl(ja, 2) == Load_a(0, ia))
      {
	Gupdate = g_vq_a(ia, 0)*(sysinfo.bkva/3)*cvq_a;
	Dl_new(ja, 7) = ctrl_o(ja, 7) - Gupdate;
      }
     }
  }//end of Phase A
  
  //Phase B
  for (ib = 0; ib < Llb; ++ib)
  {
    for (jb = 0; jb < Ldl; ++jb)
    {
      if (Dl(jb, 2) == Load_b(0, ib))
      {
	Gupdate = g_vq_b(ib, 0)*(sysinfo.bkva / 3)*cvq_b;
	Dl_new(jb, 9) = ctrl_o(jb, 9) - Gupdate;
      }
    }
   }//end of Phase B
   
   //Phase C
  for (ic = 0; ic < Llc; ++ic)
  {
    for (jc = 0; jc < Ldl; ++jc)
    {
      if (Dl(jc, 2) == Load_c(0, ic))
      {
	Gupdate = g_vq_c(ic, 0)*(sysinfo.bkva / 3)*cvq_c;
	Dl_new(jc, 11) = ctrl_o(jc, 11) - Gupdate;
      }
     }
  }// end of Phase C
  
mat Dl_nsize = Dl_new;  

dpf_re = DPF_return3(Dl_nsize, Z);// run DPF based on new step-size
Vpolar = dpf_re.Vpolar;
PQb = dpf_re.PQb;
PQL = dpf_re.PQL;


Pla_t = sum(PQL.col(0));
Plb_t = sum(PQL.col(2));
Plc_t = sum(PQL.col(4));

// calculate power loss
mat Ploss_nph;
double Ploss_nsize;
Pload_total<< Pla_t << Plb_t << Plc_t << endr;
Ploss_nph << (PQb(0, 0) - Pla_t) << (PQb(0, 2) - Plb_t) << (PQb(0, 4) - Plc_t) << endr;
Ploss_nsize = accu(Ploss_nph);
//cout << "total loss (kW):" << Ploss_nsize << endl;

if (Ploss_nsize>Ploss_osize)
{
  Dl = Dl_osize;//save the Dl based on old size
  Ploss_aftter_ctrl = Ploss_osize; 
  du;
  cout << " Best step-size obtained! Searching ends at the " << m+1 << "th iteration \n";
  cout << " Expected power loss (kW) = " << Ploss_aftter_ctrl << endl;
  cout << " Expected power loss reduction (kW) = " << Ploss_orig - Ploss_aftter_ctrl << endl;
  step_size;
 
  // send messages to slaves
if (Ploss_osize < Ploss_orig)// grad message will NOT be sent to slaves if loss is not reduced
{
  BOOST_FOREACH(CPeerNode peer, m_peers | boost::adaptors::map_values)//for broadcast use
        {
      	    
	    //ModuleMessage mm = VoltageDelta(2, 3.0, "NCSU");
            //peer.Send(mm);
	    
	    mat S2;
	    S2 << Dl(0,7) << Dl(1,7) << Dl(2,7) << endr;
	    S2 = S2.t();
	    if(peer.GetUUID() != GetUUID())
	    {
	    	ModuleMessage mg = Gradient(S2);
	    	peer.Send(mg);
	    }
	}
	
}
else
{
}
  
  break;
}
else
{
 Ploss_aftter_ctrl = Ploss_osize; 
}
  
if ( m == m_max )
{
  cout << " Unable to obtain the best step-size withing " << m+1 << "th iterations \n";
  break;
}

if ( Ploss_aftter_ctrl > Ploss_orig )
{
  cout << " Direction of gradients need to be reversed! \n";
  flag = false;
  Vmin = Vmin_orig;
  Vmax = Vmax_orig;
}
else
{ 
}



}// end of step-size search

if (flag == false)
{
  cvq_a=-beta0/(sysinfo.bkva/3)/gabs_min; 
  cvq_b=-beta0/(sysinfo.bkva/3)/gabs_min;
  cvq_c=-beta0/(sysinfo.bkva/3)/gabs_min;
  flag = true;
  
for ( int m = 0; m < m_max; m++ )
{
  // update Qinj for three phase separately
  double Gupdate;
  //Phase A
  for (ia = 0; ia < Lla; ++ia)
  {
    for (ja = 0; ja < Ldl; ++ja)
    {
      if (Dl(ja, 2) == Load_a(0, ia))
      {
	Gupdate = g_vq_a(ia, 0)*(sysinfo.bkva/3)*cvq_a;
	Dl_new(ja, 7) = ctrl_o(ja, 7) - Gupdate;
      }
     }
  }//end of Phase A
  
  //Phase B
  for (ib = 0; ib < Llb; ++ib)
  {
    for (jb = 0; jb < Ldl; ++jb)
    {
      if (Dl(jb, 2) == Load_b(0, ib))
      {
	Gupdate = g_vq_b(ib, 0)*(sysinfo.bkva / 3)*cvq_b;
	Dl_new(jb, 9) = ctrl_o(jb, 9) - Gupdate;
      }
    }
   }//end of Phase B
   
   //Phase C
  for (ic = 0; ic < Llc; ++ic)
  {
    for (jc = 0; jc < Ldl; ++jc)
    {
      if (Dl(jc, 2) == Load_c(0, ic))
      {
	Gupdate = g_vq_c(ic, 0)*(sysinfo.bkva / 3)*cvq_c;
	Dl_new(jc, 11) = ctrl_o(jc, 11) - Gupdate;
      }
     }
  }// end of Phase C
mat Dl_osize = Dl_new;  
mat du_temp=Dl_osize-ctrl_o;
du = du_temp.cols(6,11);

//cout << "Dl_new = \n" << Dl_new << endl;
// DPF based on Dl_new

VPQ dpf_re = DPF_return3(Dl_osize, Z);
//cout << "Vpolar = \n" << dpf_re.Vpolar << endl;
mat Vpolar = dpf_re.Vpolar;
mat PQb = dpf_re.PQb;
mat PQL = dpf_re.PQL;

int Lvp = Vpolar.n_rows;
int Wvp = Vpolar.n_cols;
int Lpq = PQb.n_rows;
int Wpq = PQb.n_cols;
double Pla_t = sum(PQL.col(0));
double Plb_t = sum(PQL.col(2));
double Plc_t = sum(PQL.col(4));

// calculate power loss
mat Pload_total, Ploss_oph;
double Ploss_osize;
Pload_total<< Pla_t << Plb_t << Plc_t << endr;
Ploss_oph << (PQb(0, 0) - Pla_t) << (PQb(0, 2) - Plb_t) << (PQb(0, 4) - Plc_t) << endr;
Ploss_osize = accu(Ploss_oph);
//cout << "total load (kW) per phase:" << Pload_total << endl;
//cout << "total loss (kW):" << Ploss_osize << endl;

Vabc V_abc = V_abc_list(Vpolar,Node_f,Lvp,Lnum_a,Lnum_b,Lnum_c);


mat V_a = V_abc.V_a;
mat V_b = V_abc.V_b;
mat V_c = V_abc.V_c;

mat Vmin_abc, Vmax_abc;
Vmin_abc << min(min(V_a)) << min(min(V_b)) << min(min(V_c))<<endr;
Vmax_abc << max(max(V_a)) << max(max(V_b)) << max(max(V_c))<<endr;
Vmin = min(min(Vmin_abc));
Vmax = max(max(Vmax_abc));


step_size << cvq_a << cvq_b << cvq_c << endr;
cout << "\n \n step-size at the " << m+1 << "th iteration = " << step_size(0,0) << endl;

// new step-size
cvq_a=alpha*cvq_a;
cvq_b=alpha*cvq_b;
cvq_c=alpha*cvq_c; 

//update Dl_new based on the new step-size
//Phase A
  for (ia = 0; ia < Lla; ++ia)
  {
    for (ja = 0; ja < Ldl; ++ja)
    {
      if (Dl(ja, 2) == Load_a(0, ia))
      {
	Gupdate = g_vq_a(ia, 0)*(sysinfo.bkva/3)*cvq_a;
	Dl_new(ja, 7) = ctrl_o(ja, 7) - Gupdate;
      }
     }
  }//end of Phase A
  
  //Phase B
  for (ib = 0; ib < Llb; ++ib)
  {
    for (jb = 0; jb < Ldl; ++jb)
    {
      if (Dl(jb, 2) == Load_b(0, ib))
      {
	Gupdate = g_vq_b(ib, 0)*(sysinfo.bkva / 3)*cvq_b;
	Dl_new(jb, 9) = ctrl_o(jb, 9) - Gupdate;
      }
    }
   }//end of Phase B
   
   //Phase C
  for (ic = 0; ic < Llc; ++ic)
  {
    for (jc = 0; jc < Ldl; ++jc)
    {
      if (Dl(jc, 2) == Load_c(0, ic))
      {
	Gupdate = g_vq_c(ic, 0)*(sysinfo.bkva / 3)*cvq_c;
	Dl_new(jc, 11) = ctrl_o(jc, 11) - Gupdate;
      }
     }
  }// end of Phase C
  
mat Dl_nsize = Dl_new;  

dpf_re = DPF_return3(Dl_nsize, Z);// run DPF based on new step-size
Vpolar = dpf_re.Vpolar;
PQb = dpf_re.PQb;
PQL = dpf_re.PQL;


Pla_t = sum(PQL.col(0));
Plb_t = sum(PQL.col(2));
Plc_t = sum(PQL.col(4));

// calculate power loss
mat Ploss_nph;
double Ploss_nsize;
Pload_total<< Pla_t << Plb_t << Plc_t << endr;
Ploss_nph << (PQb(0, 0) - Pla_t) << (PQb(0, 2) - Plb_t) << (PQb(0, 4) - Plc_t) << endr;
Ploss_nsize = accu(Ploss_nph);
cout << "total loss (kW):" << Ploss_nsize << endl;

if (Ploss_nsize>Ploss_osize)
{
  Dl = Dl_osize;//save the Dl based on old size
  Ploss_aftter_ctrl = Ploss_osize; 
  du;
  cout << " Best step-size obtained! Searching ends at the " << m+1 << "th iteration \n";
  cout << " Expected power loss (kW) = " << Ploss_aftter_ctrl << endl;
  cout << " Expected power loss reduction (kW) = " << Ploss_orig - Ploss_aftter_ctrl << endl;
  step_size;
  
  // send messages to slaves
if (Ploss_osize < Ploss_orig)// grad message will NOT be sent to slaves if loss is not reduced
{ 
  BOOST_FOREACH(CPeerNode peer, m_peers | boost::adaptors::map_values)//for broadcast use
        {
      	    
	    //ModuleMessage mm = VoltageDelta(2, 3.0, "Gradient sent!");
            //peer.Send(mm);
	    	   
	    mat S2;
	    S2 << Dl(0,7) << Dl(1,7) << Dl(2,7) << endr;
	    S2 = S2.t();
	    if (peer.GetUUID() != GetUUID())
	    {
	    	ModuleMessage mg = Gradient(S2);
	    	peer.Send(mg);
	    }
	}

}
else
{
}
	
  break;
}
else
{
 Ploss_aftter_ctrl = Ploss_osize; 
}
  
if ( m == m_max )
{
  cout << " Unable to obtain the best step-size withing " << m+1 << "th iterations \n";
  break;
}

if ( Ploss_aftter_ctrl > Ploss_orig )
{
  cout << " Gradient VVC failed! Both directions cannot reduce power loss \n";
  Vmin = Vmin_orig;
  Vmax = Vmax_orig;
}
else
{
}

}// end of step-size search
  
}// end of if flag



}// end of vvc_main()


}//namespace vvc
}// namespace broker
}// namespace freedm
			
