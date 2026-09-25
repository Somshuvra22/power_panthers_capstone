KPI:                      Unplanned stoppages per month: current 15/month; target: reduce from current level and recover availability toward the earlier ~94% baseline (exact numeric target to be agreed with operations)
Business problem:         Unplanned stoppages have increased from 7 to 15 per month while availability has fallen from 94.0% to 90.7%. Most stops show a detectable sensor precursor, creating an opportunity for earlier maintenance intervention.
ML problem (Version 2):   Using rolling 2-hour vibration, temperature, current and load trends, detect abnormal operating windows ahead of a likely stoppage (recall/F1) so maintenance can be alerted with roughly two hours’ lead time — recognising this will catch the majority, not all, unplanned stops, since about a quarter show no prior signal change.
ML type:                  anomaly detection
Target:                   Abnormal operating window within a 2-hour lead time before a stop
Inputs (features):        Rolling vibration, temperature, current and load trend over the prior 2 hours
Business success metric:  Reduce monthly unplanned-stop count from 15/month and recover availability toward the earlier ~94% baseline; exact numeric target to be agreed with operations
Model success metric:     Recall and F1; recall prioritised because missing a real precursor is the costlier error
Track:                    Table (Process)
Data used in class:       SYNTHETIC, generated to match our Day-1 evidence
