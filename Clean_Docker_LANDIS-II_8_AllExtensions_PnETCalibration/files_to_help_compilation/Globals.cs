using Landis.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Landis.Library.PnETCohorts
{
    public static class Globals
    {
        public static ICore ModelCore;
        public static ushort IMAX { get; private set; }
        public static DateTime StartDate {  get; private set; }
        public static int MinSpinUpClimateYear { get; private set; }
        public static int MaxSpinUpClimateYear { get; private set; }
        public static int MinFutureClimateYear { get; private set; }
        public static int MaxFutureClimateYear { get; private set; }
        public static int MaxSpinUpIndex {  get; private set; }
        public static int MaxFutureClimateIndex {  get; private set; }
        public static readonly object CWDThreadLock = new object();
        public static readonly object litterThreadLock = new object();
        public static readonly object distributionThreadLock = new object();
        public static readonly object ecoregionDataThreadLock = new object();
        public static readonly object initialSitesThreadLock = new object();
        public const float bulkIntercept = 165.0f; //kg/m3
        public const float bulkSlope = 1.3f; //kg/m3
        public const float Pwater = 1000.0f;  // Density of water (kg/m3)
        public const float lambAir = 0.023f; // W/m K (CLM5 documentation, Table 2.7)
        public const float lambIce = 2.29f; // W/m K (CLM5 documentation, Table 2.7)
        public const float snowHeatCapacity = 2090f; // J/kg K (https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html)
        public const float snowReflectanceThreshold = 0.100f;  // minimum depth of snow (m) that counts as full snow for albedo calculations

        public static void InitializeCore(ICore mCore, ushort IMAX, DateTime startDate)
        {
            Globals.StartDate = startDate;
            ModelCore = mCore;
            Globals.IMAX = IMAX;
        }

        public static void SetMinMaxClimateYears()
        {
            MinSpinUpClimateYear = Climate.Climate.SpinupCalendarYear(1);
            MaxSpinUpClimateYear = Climate.Climate.SpinupEcoregionYearClimate.First(x => x != null).Last(x => x!= null).CalendarYear;

            MinFutureClimateYear = Climate.Climate.FutureCalendarYear(1);
            MaxSpinUpClimateYear = Climate.Climate.FutureEcoregionYearClimate.First(x => x != null).Last(x => x != null).CalendarYear;
			ModelCore.UI.WriteLine("Climate.Climate.FutureCalendarYear(2) :");
			ModelCore.UI.WriteLine(Climate.Climate.FutureCalendarYear(2).ToString());
			ModelCore.UI.WriteLine("MaxSpinUpClimateYear :");
			ModelCore.UI.WriteLine(MaxSpinUpClimateYear.ToString());
        }

        public static bool IsFutureClimate(DateTime date)
        {
            if (date.Year - MinFutureClimateYear + 1 <= 0)
            {
                return false;
            }
            return true;
        }

        public static int ConvertYearToFutureClimateYear(DateTime date)
        {
			ModelCore.UI.WriteLine("date.Year :");
			ModelCore.UI.WriteLine(date.Year.ToString());
			ModelCore.UI.WriteLine("MinFutureClimateYear :");
			ModelCore.UI.WriteLine(MinFutureClimateYear.ToString());
            int convert = date.Year - MinFutureClimateYear + 1;

            return convert >= 1 ? convert : -1;
        }

        public static int ConvertYearToSpinUpClimateYear(DateTime date)
        {
			ModelCore.UI.WriteLine("date.Year :");
			ModelCore.UI.WriteLine(date.Year.ToString());
			ModelCore.UI.WriteLine("MinSpinUpClimateYear :");
			ModelCore.UI.WriteLine(MinSpinUpClimateYear.ToString());
            int convert = date.Year - MinSpinUpClimateYear + 1;

            return convert >= 1 ? convert : -1;
        }
    }
}
