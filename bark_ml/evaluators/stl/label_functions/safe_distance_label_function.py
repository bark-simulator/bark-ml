import sys
from bark.core.world.evaluation.ltl import SafeDistanceLabelFunction
from rtamt.spec.stl.discrete_time.specification import StlDiscreteTimeSpecification
import logging
import rtamt
from bark_ml.evaluators.stl.label_functions.base_label_function import BaseQuantizedLabelFunction

class SafeDistanceQuantizedLabelFunction(SafeDistanceLabelFunction, BaseQuantizedLabelFunction):      
    robustness_min = float('inf')  
    robustness_max = float('-inf')  

    lon_value_range = (-200, 200)
    lat_value_range = (-20, 20)
    feature_range = (-1, 1)
    
    def __init__(self, label_str: str, to_rear: bool, delta_ego: float, delta_others: float, a_e: float, a_o: float, 
                 consider_crossing_corridors: bool, max_agents_for_crossing: int, use_frac_param_from_world: bool, 
                 lateral_difference_threshold: float, angle_difference_threshold: float, check_lateral_dist: bool, 
                 signal_sampling_period: float = 200, robustness_normalized: bool = True):        
        super().__init__(label_str, to_rear, delta_ego, delta_others, a_e, a_o, consider_crossing_corridors, 
                         max_agents_for_crossing, use_frac_param_from_world, lateral_difference_threshold, 
                         angle_difference_threshold, check_lateral_dist)
        self.initialize_specs(signal_sampling_period)     

        self.robustness_lon = float('-inf')
        self.robustness_lat = float('-inf')
        self.robustness = float('-inf')
        self.robustness_normalized = robustness_normalized

    def initialize_specs(self, signal_sampling_period: float):
        self.stl_spec_timestep = 0
        self.stl_spec_lon_checked = False      
        self.stl_spec_lat_checked = False   

        self.stl_spec_lon = StlDiscreteTimeSpecification()
        self.stl_spec_lon.declare_var('dist', 'float')
        self.stl_spec_lon.declare_var('safe_dist_0', 'float')
        self.stl_spec_lon.declare_var('safe_dist_1', 'float')   
        self.stl_spec_lon.declare_var('safe_dist_2', 'float')
        self.stl_spec_lon.declare_var('safe_dist_3', 'float')   
        self.stl_spec_lon.declare_var('delta', 'float')
        self.stl_spec_lon.declare_var('t_stop_f', 'float')   
        self.stl_spec_lon.declare_var('t_stop_f_star', 'float')   
        self.stl_spec_lon.declare_var('a_f', 'float')   
        self.stl_spec_lon.declare_var('a_r', 'float')   
        self.stl_spec_lon.declare_var('v_f_star', 'float')   
        self.stl_spec_lon.declare_var('v_r', 'float')   
        self.stl_spec_lon.declare_var('t_stop_r', 'float')   

        self.stl_spec_lon.unit = 's'
        self.stl_spec_lon.set_sampling_period(signal_sampling_period, 'ms', 0.1)

        formula_lon = "(dist < 0.0)" \
        + ' or ((dist > safe_dist_0 or (delta <= t_stop_f and dist > safe_dist_3))' \
        + ' or ((delta <= t_stop_f and a_f > a_r and v_f_star < v_r and t_stop_r < t_stop_f_star)) and (dist > safe_dist_2))' \
        + ' or (dist > safe_dist_1)'

        self.stl_spec_lon.spec = formula_lon
            
        try:
            self.stl_spec_lon.parse()
            self.stl_spec_lon.pastify()
        except rtamt.RTAMTException as err:
            logging.info('RTAMT Exception: {}'.format(err))
            sys.exit()

        self.stl_spec_lat = StlDiscreteTimeSpecification()
        self.stl_spec_lat.declare_var('dist_lat', 'float')
        self.stl_spec_lat.declare_var('lateral_positive', 'float')
        self.stl_spec_lat.declare_var('v_1_lat', 'float')
        self.stl_spec_lat.declare_var('v_2_lat', 'float')
        self.stl_spec_lat.declare_var('min_lat_safe_dist', 'float')

        self.stl_spec_lat.unit = 's'
        self.stl_spec_lat.set_sampling_period(signal_sampling_period, 'ms', 0.1)

        formula_lat = 'dist_lat !== 0.0 and' \
            + ' ((v_1_lat >= 0.0 and v_2_lat <= 0.0 and dist_lat < 0.0)' \
            + ' or (v_1_lat <= 0.0 and v_2_lat >= 0.0 and dist_lat > 0.0)' \
            + ' or (lateral_positive > min_lat_safe_dist))'

        self.stl_spec_lat.spec = formula_lat
        
        try:
            self.stl_spec_lat.parse()
            self.stl_spec_lat.pastify()
        except rtamt.RTAMTException as err:
            logging.info('RTAMT Exception: {}'.format(err))
            sys.exit()

        logging.info("Successfully parsed the SD STL formulas")

    def compute_robustness(self, eval_result):        
        safe_distance = eval_result 

        if (self.stl_spec_lon_checked):
            self.robustness_lon = self.normalize_robustness(self.robustness_lon, self.feature_range, self.lon_value_range) 

        if (self.stl_spec_lat_checked):
            self.robustness_lat = self.normalize_robustness(self.robustness_lat, self.feature_range, self.lat_value_range) 

        if (not self.stl_spec_lon_checked and not self.stl_spec_lat_checked):
            if safe_distance:
                if SafeDistanceQuantizedLabelFunction.robustness_max >= 0.0:   
                    self.robustness = SafeDistanceQuantizedLabelFunction.robustness_max
                else:                    
                    self.robustness = max(self.feature_range)
            else:
                if SafeDistanceQuantizedLabelFunction.robustness_min <= 0.0:   
                    self.robustness = SafeDistanceQuantizedLabelFunction.robustness_min
                else:    
                    self.robustness = min(self.feature_range)
        elif (self.stl_spec_lon_checked and self.stl_spec_lat_checked):            
            if safe_distance and (self.robustness_lon < 0.0 or self.robustness_lat < 0.0):
                self.robustness = max(self.robustness_lon, self.robustness_lat)   
            else:    
                self.robustness = min(self.robustness_lon, self.robustness_lat)
        elif (self.stl_spec_lon_checked):
            self.robustness = self.robustness_lon

        # logging.info(f"Current robustness for SD={self.robustness}")

        if self.robustness > SafeDistanceQuantizedLabelFunction.robustness_max:
            SafeDistanceQuantizedLabelFunction.robustness_max = self.robustness

        if self.robustness < SafeDistanceQuantizedLabelFunction.robustness_min:
            SafeDistanceQuantizedLabelFunction.robustness_min = self.robustness
        
        # logging.info(f"Current MIN robustness for SD={self.robustness_min}")
        # logging.info(f"Current MAX robustness for SD={self.robustness_max}")

    def normalize_robustness(self, robustness, feature_range, value_range):
        if self.robustness_normalized: 
            # logging.info(f"Robustness BEFORE Normalization={robustness} and value_range={value_range}")
            min_value, max_value = value_range
            min_range, max_range = feature_range
        
            scaled_robustness = ((robustness - min_value) / (max_value - min_value)) * (max_range - min_range) + min_range

            # logging.info(f"Robustness AFTER Normalization={scaled_robustness}")

            return scaled_robustness
        else:
            return robustness

    def Evaluate(self, observed_world):
        self.stl_spec_timestep = observed_world.time
        self.stl_spec_lon_checked = False      
        self.stl_spec_lat_checked = False   

        eval_result = super().Evaluate(observed_world)

        self.compute_robustness(next(iter(eval_result.values())))

        return eval_result
    
    def CheckSafeDistanceLongitudinal(self, v_f: float, v_r: float, dist: float, a_r: float,  a_f: float, delta: float):
        self.stl_spec_lon_checked = True

        v_f_star = self.CalcVelFrontStar(v_f, a_f, delta)
        t_stop_f_star = -v_f_star / a_r
        t_stop_r = -v_r / a_r
        t_stop_f = -v_f / a_f

        ZeroToPositive = lambda safe_dist: 0.0 if safe_dist < 0.0 else safe_dist
        safe_dist_0 = ZeroToPositive(self.CalcSafeDistance0(v_r, a_r, delta))
        safe_dist_1 = ZeroToPositive(self.CalcSafeDistance1(v_r, v_f, a_r, a_f, delta))
        safe_dist_2 = ZeroToPositive(self.CalcSafeDistance2(v_r, v_f, a_r, a_f, delta))
        safe_dist_3 = ZeroToPositive(self.CalcSafeDistance3(v_r, v_f, a_r, a_f, delta))
        # logging.info(f"sf0={safe_dist_0}, sf1={safe_dist_1}, sf2={safe_dist_2}, sf3={safe_dist_3}")

        # Updating STL monitor
        self.robustness_lon = self.stl_spec_lon.update(self.stl_spec_timestep, [('dist', dist),
                                                     ('safe_dist_0', safe_dist_0),
                                                     ('safe_dist_1', safe_dist_1),
                                                     ('safe_dist_2', safe_dist_2),
                                                     ('safe_dist_3', safe_dist_3),
                                                     ('delta', delta),
                                                     ('t_stop_f', t_stop_f),
                                                     ('t_stop_f_star', t_stop_f_star),
                                                     ('a_f', a_f),
                                                     ('a_r', a_r),
                                                     ('v_f_star', v_f_star),
                                                     ('v_r', v_r),
                                                     ('t_stop_r', t_stop_r)])
        # logging.info(f"CheckSafeDistanceLongitudinal: Robustness STL spec result in the label function: {self.robustness_lon}")
            
        safe_distance_lon = self.robustness_lon > 0.0

        if self.robustness_lon == 0.0:
            safe_distance_lon = super().CheckSafeDistanceLongitudinal(v_f, v_r, dist, a_r,  a_f, delta)

        return safe_distance_lon

    def CheckSafeDistanceLateral(self, v_1_lat: float, v_2_lat: float, dist_lat: float, a_1_lat: float,  a_2_lat: float, delta1: float, delta2: float):
        # return super().CheckSafeDistanceLateral(v_1_lat, v_2_lat, dist_lat, a_1_lat, a_2_lat, delta1, delta2)    
        self.stl_spec_lat_checked = True        

        # For convention of RSS paper, make v_1_lat be larger (e.g. positive compared to v_2_lat) ...
        v_1_lat_orig = v_1_lat
        v_2_lat_orig = v_2_lat

        if v_1_lat < v_2_lat:
            v_1_lat, v_2_lat = v_2_lat, v_1_lat
            delta1, delta2 = delta2, delta1
            a_1_lat, a_2_lat = a_2_lat, a_1_lat

        # ... lateral distance positive
        lateral_positive = abs(dist_lat)

        min_lat_safe_dist = (
            v_1_lat * delta1 +
            (v_1_lat * delta1 if v_1_lat == 0.0 else v_1_lat * v_1_lat / (2 * a_1_lat)) -
            (v_2_lat * delta2 - (v_2_lat * delta2 if v_2_lat == 0.0 else v_2_lat * v_2_lat / (2 * a_2_lat)))
        )
        # logging.info("Min lat safe dist:", min_lat_safe_dist)

        # Updating STL monitor
        self.robustness_lat = self.stl_spec_lat.update(self.stl_spec_timestep, [('dist_lat', dist_lat),
                                                     ('lateral_positive', lateral_positive),
                                                     ('v_1_lat', v_1_lat_orig),
                                                     ('v_2_lat', v_2_lat_orig),
                                                     ('min_lat_safe_dist', min_lat_safe_dist)
                                                     ])
        # logging.info(f"CheckSafeDistanceLateral: Robustness STL spec result in the label function: {self.robustness_lat}")

        safe_distance_lat = self.robustness_lat > 0.0

        if self.robustness_lat == 0.0:
            safe_distance_lat = super().CheckSafeDistanceLateral(v_1_lat, v_2_lat, dist_lat, a_1_lat,  a_2_lat, delta1, delta2)

        return safe_distance_lat